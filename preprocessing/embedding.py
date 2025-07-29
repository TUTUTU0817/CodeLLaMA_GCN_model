import argparse
import gc
import json
import os
import pickle
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import yaml
import utils 
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.nn import Embedding
import logging


# === 設定區 ===
TASK = "multi"  # 可選 "multi" 或 "binary"
BASE_DIR = "/home/tu/exp/EXP_final/Example_output"
GRAPH_DIR = os.path.join(BASE_DIR, "graph")
DATA_JSON_PATH = os.path.join(BASE_DIR, "result.jsonl")
LOG_PATH = os.path.join(BASE_DIR, "output_graph_embedding.log")
LLM_PATH = "/home/tu/exp/EXP_final/finetune/result/QLoRA_Codellama_classification_7b_3/checkpoint-1797"

# === 設定 logging 基本配置 ===
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_PATH, mode="a"),
                              logging.StreamHandler()])

# === 輔助函式 ===
class OptimizedNodesEmbedding:
    def __init__(self, nodes_dim: int, codellama_model, codellama_tokenizer):
        self.codellama_model = codellama_model
        self.codellama_tokenizer = codellama_tokenizer
        # self.tokenizer_bert = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        # self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to("cuda")
        self.nodes_dim = nodes_dim
        
        # 加入程式碼快取
        self.code_cache = {}
        self.max_cache_size = 1000  # 減少快取大小避免記憶體過大
        
        assert self.nodes_dim >= 0

    def __call__(self, nodes, type_embedding):
        embedded_nodes, types, codes, lines = self.embed_nodes_batch(nodes, type_embedding)
        nodes_tensor = torch.stack(embedded_nodes)
        return nodes_tensor, types, codes, lines

    def get_code_embedding_cached(self, code_snippet):
        """使用快取的程式碼嵌入"""
        code_hash = hash(code_snippet)
        
        if code_hash in self.code_cache:
            return self.code_cache[code_hash].clone()
        
        # 如果快取太大，清理一部分
        if len(self.code_cache) > self.max_cache_size:
            keys_to_remove = list(self.code_cache.keys())[:self.max_cache_size//2]
            for key in keys_to_remove:
                del self.code_cache[key]
        
        # 計算嵌入
        inputs = self.codellama_tokenizer(
            code_snippet, 
            return_tensors="pt", 
            max_length=512, 
            padding=True, 
            truncation=True
        ).to(utils.device)
        
        with torch.no_grad():
            outputs = self.codellama_model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
        
        # 快取結果
        self.code_cache[code_hash] = embedding.clone()
        return embedding

    def embed_nodes_batch(self, nodes, type_embedding, batch_size=8):
        """批次處理節點嵌入"""
        embeddings = []
        types = []
        codes = []
        lines = []
        
        # 收集所有節點資訊
        node_data = []
        for n_id, node in nodes.items():
            node_data.append({
                'code': node['code'],
                'type': node['type'],
                'line': node['lineNumber']
            })
        
        # 批次處理
        for i in range(0, len(node_data), batch_size):
            batch_data = node_data[i:i+batch_size]
            batch_codes = [item['code'] for item in batch_data]
            
            # 檢查快取
            batch_embeddings = []
            uncached_indices = []
            uncached_codes = []
            
            for j, code in enumerate(batch_codes):
                code_hash = hash(code)
                if code_hash in self.code_cache:
                    batch_embeddings.append(self.code_cache[code_hash].clone())
                else:
                    uncached_indices.append(j)
                    uncached_codes.append(code)
                    batch_embeddings.append(None)  # 佔位符
            
            # 批次處理未快取的程式碼
            if uncached_codes:
                inputs = self.codellama_tokenizer(
                    uncached_codes,
                    return_tensors="pt",
                    max_length=512,
                    padding=True,
                    truncation=True
                ).to(utils.device)
                
                with torch.no_grad():
                    outputs = self.codellama_model(**inputs, output_hidden_states=True)
                    new_embeddings = outputs.hidden_states[-1].mean(dim=1)
                
                # 更新快取和批次結果
                for idx, uncached_idx in enumerate(uncached_indices):
                    embedding = new_embeddings[idx]
                    code_hash = hash(uncached_codes[idx])
                    self.code_cache[code_hash] = embedding.clone()
                    batch_embeddings[uncached_idx] = embedding
            
            # 處理類型嵌入並組合
            for j, (code_emb, item) in enumerate(zip(batch_embeddings, batch_data)):
                type_index = utils.node_type_dict.get(item['type'], utils.node_type_dict["UNKNOWN"])
                type_emb = type_embedding(torch.tensor(type_index, dtype=torch.long, device=utils.device))
                
                embedding = torch.cat((type_emb, code_emb), dim=0)
                embeddings.append(embedding)
                types.append(item['type'])
                codes.append(item['code'])
                lines.append(item['line'])
        
        return embeddings, types, codes, lines

    
class GraphsEmbedding:
    def __init__(self, edge_type: list):
        self.edge_type = edge_type
        self.edge_type_set = set(edge_type)  # 使用 set 提高查找效率

    def __call__(self, nodes, edges, edge_embedding):
        connections, edge_embeddings, edge_types = self.nodes_connectivity(nodes, edges, edge_embedding)
        if connections[0]:
            return torch.tensor(connections, dtype=torch.long), torch.stack(edge_embeddings), edge_types
        else:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, edge_embedding.embedding_dim), dtype=torch.float), []

    def nodes_connectivity(self, nodes, edges, edge_embedding):
        """優化的節點連接處理"""
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes.keys())}
        node_ids = set(nodes.keys())  # 使用 set 提高查找效率

        coo = [[], []]
        edge_embeddings = []
        edge_types = []

        # 預先過濾並批次處理邊
        valid_edges = []
        for edge_group in edges:
            for edge in edge_group:
                if (edge.get('type') in self.edge_type_set and 
                    edge.get("source") in node_ids and 
                    edge.get("target") in node_ids):
                    valid_edges.append(edge)

        # 批次處理邊的嵌入
        if valid_edges:
            edge_types_batch = [edge.get("type") for edge in valid_edges]
            edge_indices = [utils.edge_type_dict.get(edge_type, 0) for edge_type in edge_types_batch]
            # 批次計算邊嵌入
            edge_indices_tensor = torch.tensor(edge_indices, dtype=torch.long, device=utils.device)
            edge_embeddings_batch = edge_embedding(edge_indices_tensor)
            
            # 建立連接
            for i, edge in enumerate(valid_edges):
                source_idx = node_id_to_idx[edge.get("source")]
                target_idx = node_id_to_idx[edge.get("target")]
                
                coo[0].append(source_idx)
                coo[1].append(target_idx)
                edge_embeddings.append(edge_embeddings_batch[i])
                edge_types.append(edge.get("type"))
        return coo, edge_embeddings, edge_types


def nodes_to_input_optimized(model, tokenizer, nodes, edges, target, nodes_dim, edge_type, type_embedding, edge_embedding):
    """優化的節點轉輸入函數"""
    nodes_embedding = OptimizedNodesEmbedding(nodes_dim, model, tokenizer)
    graphs_embedding = GraphsEmbedding(edge_type)
    
    x, types, codes, lines = nodes_embedding(nodes, type_embedding)
    edge_index, edge_attr, edge_types = graphs_embedding(nodes, edges, edge_embedding)
    # 直接在 GPU 上創建，最後再移到 CPU
    label = torch.tensor([target], dtype=torch.long, device=utils.device)
    
    pyg_data = Data(
        x=x.cpu().detach(), 
        edge_index=edge_index.cpu().detach(), 
        edge_attr=edge_attr.cpu().detach(), 
        y=label.cpu().detach(), 
        types=types, 
        codes=codes,
        edge_type=edge_types,
        line=lines
    )
    return pyg_data

# === 主要函數 ===
def collect_by_func_for_multi_optimized(model, tokenizer, json_file_name, graph, edge_type, type_embedding, edge_embedding, data_dict):
    """優化的多分類收集函數"""
    nodes_dim = model.config.hidden_size + 64
    method_nodes = graph["nodes"]
    edges = graph["edges"]
    
    for method_group in method_nodes:
        # 第一遍：收集所有有效的方法
        valid_methods = []
        for i, node_info in enumerate(method_group):
            if node_info["type"] == "METHOD":
                method_full_name = node_info["fullName"].replace("/", "_").replace(":", "_").replace("(", "_").replace(")", "_").replace(",", "_").replace(" ", "_")
                method_line_number = node_info["lineNumber"]
                key = f"{json_file_name}_{method_full_name}_{method_line_number}"
                if key in data_dict:
                    valid_methods.append((i, node_info, key))
        
        # 如果沒有有效方法，跳過整個 method_group
        if not valid_methods:
            continue
        
        # 第二遍：批次處理所有有效的方法
        for i, node_info, key in valid_methods:
            try:
                print(f"✅ Match found for key: {key}")
                
                # 獲取 target
                cwe_id = data_dict[key]['cwe_id'] if data_dict[key]['cwe_id'] else "normal"
                target = utils.CWE_TO_LABEL[cwe_id]
                
                # 收集節點
                method_node_list = []
                for j in range(i + 1, len(method_group)):
                    if method_group[j]["type"] == "METHOD":
                        break
                    method_node_list.append(method_group[j])
                
                if not method_node_list:
                    print(f"⚠️ Skipping empty method for key: {key}")
                    continue
                
                nodes = {node["id"]: node for node in method_node_list}
                
                pyg_data = nodes_to_input_optimized(
                    model, tokenizer, nodes, edges, target, 
                    nodes_dim, edge_type, type_embedding, edge_embedding
                )
                if pyg_data is not None:
                    data_dict[key]['full_graph'] = pyg_data
                    print(f"✅ Added full_graph for key: {key}")
                    
            except Exception as e:
                logging.error(f"Error processing method {key}: {e}")
                # 發生錯誤時也清理記憶體
                torch.cuda.empty_cache()
                gc.collect()
                continue


def load_graphs_in_chunks(graph_dir, chunk_size=20):
    """分批載入圖檔案以減少記憶體使用"""
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith(".json")]
    
    for i in range(0, len(graph_files), chunk_size):
        chunk_files = graph_files[i:i+chunk_size]
        chunk_data = []
        
        for graph_file in chunk_files:
            graph_path = os.path.join(graph_dir, graph_file)
            try:
                with open(graph_path, "r", encoding='utf-8') as f:
                    graph_data = json.load(f)
                    chunk_data.append((graph_file.replace(".json", ""), graph_data))
            except Exception as e:
                logging.error(f"Error loading {graph_file}: {e}")
                continue
        
        yield chunk_data
        
        # 清理記憶體
        del chunk_data
        gc.collect()


def main():
    
    edge_type = ["AST", "CFG", "DDG", "CDG"]

        # === 自動從 result.jsonl 建立 .pt（只會做一次）===
    if not DATA_JSON_PATH.endswith(".jsonl"):
        raise ValueError("❌ DATA_JSON_PATH 必須是 .jsonl 檔案路徑")

    data_pt_path = DATA_JSON_PATH.replace(".jsonl", ".pt")
    if not os.path.exists(data_pt_path):
        logging.info(f"📄 找不到 {data_pt_path}，將從 JSONL 建立...")
        data_pt_path = utils.convert_jsonl_to_pt(DATA_JSON_PATH)
    else:
        logging.info(f"✅ 已找到 {data_pt_path}，略過轉換")

    if TASK == "multi":
        print("🔍 載入模型...")
        model, tokenizer = utils.load_fine_tuned_model(
            num_labels=len(utils.CWE_CLASSES), 
            model_name="codellama/CodeLlama-7b-Instruct-hf", 
            checkpoint_path=LLM_PATH
        )


        # 在載入模型之後添加這些檢查
        print("🔍 檢查邊類型字典...")
        print(f"utils.edge_type_dict: {utils.edge_type_dict}")
        print(f"Expected edge types: {edge_type}")

        
        
        print("📂 載入資料...")
        data = torch.load(data_pt_path)
        data_dict = {item['key']: item for item in data}

        print("⚙️ 設定模型...")
        model = model.to(utils.device)
        model.eval()
        tokenizer = utils.setup_tokenizer(model, tokenizer)

        # 使用更小的嵌入維度以節省記憶體
        type_embedding = Embedding(len(utils.node_type_dict), 64).to(utils.device)
        edge_embedding = Embedding(len(utils.edge_type_dict), 16).to(utils.device)

        print("🚀 開始處理圖檔案...")
        total_processed = 0
        
        
         # 計算總檔案數
        print("📊 計算總檔案數...")
        total_files = len([f for f in os.listdir(GRAPH_DIR) if f.endswith(".json")])
        print(f"總共需要處理 {total_files} 個檔案")
        
        # 創建全局進度條
        overall_progress = tqdm(total=total_files, desc="Processing all graphs", unit="files")
        
        # 分批處理圖檔案
        for chunk in load_graphs_in_chunks(GRAPH_DIR, chunk_size=15):  # 減少 chunk 大小
            chunk_progress = tqdm(chunk, desc=f"Processing chunk", leave=False)
            
            for json_file_name, graph_data in chunk_progress:
                try:
                    # ✅ 每個檔案處理前檢查記憶體
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        if memory_used > 13.5:  # 如果記憶體使用超過 13.5GB
                            logging.warning(f"Memory usage high ({memory_used:.1f}GB), forcing cleanup")
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                            # 如果清理後還是太高，跳過這個檔案
                            memory_used = torch.cuda.memory_allocated() / 1024**3
                            if memory_used > 13.5:
                                logging.warning(f"Skipping {json_file_name} due to high memory usage")
                                continue
                    collect_by_func_for_multi_optimized(
                        model, tokenizer, json_file_name, graph_data,
                        edge_type, type_embedding, edge_embedding, data_dict
                    )
                    total_processed += 1
                    
                    # 更新全局進度條
                    overall_progress.update(1)
                    overall_progress.set_postfix({
                        'processed': total_processed,
                        'memory': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
                    })
                    
                    # 更頻繁的記憶體清理
                    if total_processed % 50 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logging.error(f"Error processing {json_file_name}: {e}")
                    continue
            
            # 每個 chunk 處理完後清理記憶體
            gc.collect()
            torch.cuda.empty_cache()

        print("💾 同步更新資料...")
        # 同步更新
        updated_count = 0
        for item in tqdm(data, desc="Syncing data"):
            key = item['key']
            if key in data_dict and 'full_graph' in data_dict[key]:
                item['full_graph'] = data_dict[key]['full_graph']
                updated_count += 1

        print(f"✅ 更新了 {updated_count} 個項目")

        # 保存結果
        original_with_fullgraph = data_pt_path.replace('.pt', '_with_fullgraph.pt')
        torch.save(data, original_with_fullgraph)
        print(f"Updated data saved to {original_with_fullgraph}")                
        
        # 最終清理
        del data, data_dict, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 檢查 GPU 狀態
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")   
    main()