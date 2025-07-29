import gc
import json
import os
import random
import time
import pandas as pd
import re
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import pickle
import logging
import psutil

# ==== 路徑設定區 ====
BASE_DIR = "/home/tu/exp/EXP_final/Dataset/multi"
# TRAIN_PT = os.path.join(BASE_DIR, "vul_train.pt")
# EVAL_PT = os.path.join(BASE_DIR, "vul_eval.pt")
# TEST_PT = os.path.join(BASE_DIR, "vul_test.pt")
# TRAIN_OUT = os.path.join(BASE_DIR, "vul_train.pt")
# EVAL_OUT = os.path.join(BASE_DIR, "vul_eval.pt")
# TEST_OUT = os.path.join(BASE_DIR, "vul_test.pt")
EXAMPLE_PT = "/home/tu/exp/EXP_final/Example_output/result_with_fullgraph.pt"
EXAMPLE_OUT_PT = "/home/tu/exp/EXP_final/Example_output/result_with_fullgraph.pt"
TMP_DIR = "/home/tu/exp/EXP_final/preprocessing/graph_processing"
LOG_PATH = "/home/tu/exp/EXP_final/Example_output/output_graph_process_sliced.log"
SLICE_MODE = "buggy_lines"  # 可選 "buggy_lines" 或 "node_type"
# ==== 設定 logging 基本配置 ====
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_PATH, mode="a"),
                              logging.StreamHandler()])
# ==== 輔助函示 ====
def log_memory_usage():
    """記錄記憶體使用情況"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    logging.info(f"🧠 記憶體使用: {mem_mb:.2f} MB")
    return mem_mb

def _build_sliced_data_optimized(data, slice_node_ids):
    """優化版本：直接使用 tensor 操作，避免 NetworkX"""
    if not slice_node_ids:
        logging.warning("⚠️ 切割後節點為空，返回空圖")
        return Data(
            x=torch.empty((0, data.x.size(1))),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, data.edge_attr.size(1))) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None,
            types=[], codes=[], edge_type=[], line=[]
        )
    
    # 轉換為列表並排序
    new_node_ids = sorted(list(slice_node_ids))
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(new_node_ids)}
    
    # 使用 tensor 操作篩選邊
    edge_index = data.edge_index
    edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src in slice_node_ids and dst in slice_node_ids:
            edge_mask[i] = True
    
    # 篩選有效邊
    valid_edges = edge_index[:, edge_mask]
    
    # 重新映射節點 ID
    new_edge_index = torch.zeros_like(valid_edges)
    for i in range(valid_edges.size(1)):
        src, dst = valid_edges[0, i].item(), valid_edges[1, i].item()
        new_edge_index[0, i] = node_id_map[src]
        new_edge_index[1, i] = node_id_map[dst]
    
    # 篩選邊的屬性
    new_edge_type = [data.edge_type[i] for i, mask in enumerate(edge_mask) if mask] if hasattr(data, 'edge_type') else []
    new_edge_attr = data.edge_attr[edge_mask] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    
    logging.info(f"⚙️ 切割後邊數量: {new_edge_index.size(1)}")
    
    # 重建 Data
    sliced_data = Data(
        x=data.x[new_node_ids],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        y=data.y if hasattr(data, 'y') else None,
        types=[data.types[i] for i in new_node_ids],
        codes=[data.codes[i] for i in new_node_ids],
        edge_type=new_edge_type,
        line=[data.line[i] for i in new_node_ids]
    )
    
    return sliced_data

def slice_graph(data, mode="buggy", target_lines=None):
    """
    通用圖切割函式。
    
    Parameters:
        data: PyG Graph 物件
        mode: "buggy" 表示用漏洞行裁切, "type" 表示用 CALL/IDENTIFIER 節點裁切
        target_lines: 僅在 mode="buggy" 時需要的參數 (list of int)
    Returns:
        裁切後的 PyG Graph
    """
    try:
        # === 1. 找出目標節點 ===
        if mode == "buggy_lines":
            if not target_lines:
                logging.warning("⚠️ Buggy 模式需提供 target_lines，返回原圖")
                return data
            target_lines_set = set(map(str, target_lines))
            target_node_ids = [i for i, ln in enumerate(data.line) if str(ln) in target_lines_set]
            if not target_node_ids:
                logging.warning("⚠️ 找不到對應漏洞行節點，返回原圖")
                return data
        elif mode == "node_type":
            target_node_ids = [i for i, t in enumerate(data.types) if t in {"CALL", "IDENTIFIER"}]
            if not target_node_ids:
                logging.warning("⚠️ 找不到 CALL/IDENTIFIER 節點，返回原圖")
                return data
        else:
            logging.error(f"❌ 不支援的裁切模式: {mode}")
            return data

        # === 2. 建立鄰接關係 ===
        edge_index = data.edge_index
        predecessors, successors, edge_types = {}, {}, {}

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_type = data.edge_type[i] if i < len(data.edge_type) else 'UNKNOWN'
            predecessors.setdefault(dst, []).append(src)
            successors.setdefault(src, []).append(dst)
            edge_types[(src, dst)] = edge_type

        # === 3. 遍歷: 找出與目標節點相連的邊 (DDG, CDG) ===
        slice_node_ids = set()
        for node_id in target_node_ids:
            stack = [node_id]
            visited = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                slice_node_ids.add(current)
                neighbors = predecessors.get(current, []) + successors.get(current, [])
                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    edge_type = edge_types.get((current, neighbor)) or edge_types.get((neighbor, current))
                    if edge_type in {"DDG", "CDG"}:
                        stack.append(neighbor)

        # === 4. 加入 AST/CFG 鄰居 ===
        additional_nodes = set()
        for node_id in slice_node_ids:
            neighbors = predecessors.get(node_id, []) + successors.get(node_id, [])
            for neighbor in neighbors:
                if neighbor in slice_node_ids:
                    continue
                edge_type = edge_types.get((node_id, neighbor)) or edge_types.get((neighbor, node_id))
                if edge_type in {"AST", "CFG"}:
                    additional_nodes.add(neighbor)

        slice_node_ids.update(additional_nodes)
        logging.info(f"⚙️ [{mode}] 切割後節點數量: {len(slice_node_ids)}")
        return _build_sliced_data_optimized(data, slice_node_ids)

    except Exception as e:
        logging.error(f"❌ [{mode}] 切割失敗: {e}")
        return data

def process_pt_dataset_memory_safe(pt_path, output_path, batch_size=10):
    """記憶體安全版本：超小批次 + 即時保存"""
    data = torch.load(pt_path)
    
    total_items = len(data)
    
    # 使用臨時檔案分段保存
    temp_dir = "/home/tu/exp/EXP_final/preprocessing/graph_processing"
    os.makedirs(temp_dir, exist_ok=True)
    
    processed_count = 0
    total_times = 0
    with tqdm(total=total_items, desc="Memory-Safe Processing") as pbar:
        start_time = time.time()
        for batch_start in range(0, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            batch_items = data[batch_start:batch_end]
            
            batch_processed = []
            
            for idx, item in enumerate(batch_items):
                # if item['cwe_id'] != 'normal' and item['cwe_id'] != '':
                #     continue
                actual_idx = batch_start + idx
                
                try:
                    # 處理單一項目

                    processed_item = process_single_item(item, actual_idx)
                    if processed_item:
                        batch_processed.append(processed_item)
                        processed_count += 1
                    
                    # 更新進度
                    pbar.update(1)
                    pbar.set_postfix({
                        '已處理': processed_count,
                        '記憶體': f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB"
                    })
                    
                except Exception as e:
                    logging.error(f"❌ 第 {actual_idx} 筆處理失敗: {e}")
                    pbar.update(1)
                    continue
            end_time = time.time()

            # # 計算並記錄處理時間
            processing_time = end_time - start_time
            total_times += processing_time
            # logging.info(f"✅ 第 {actual_idx} 筆處理完成，耗時: {processing_time:.2f}秒")

            # 保存批次結果到臨時檔案
            temp_file = os.path.join(temp_dir, f"batch_{batch_start}.pt")
            torch.save(batch_processed, temp_file)
            
            # 強制清理記憶體
            del batch_processed, batch_items
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 計算平均裁切時間
    average_slice_time = total_times / max(processed_count, 1)
    logging.info(f"📊 平均裁切時間：{average_slice_time:.4f} 秒")
    # 合併所有臨時檔案
    merge_temp_files(temp_dir, output_path)
    
    # 清理臨時檔案
    import shutil
    shutil.rmtree(temp_dir)
    
    logging.info(f"✅ 記憶體安全處理完成！處理了 {processed_count} 筆資料，時間: {total_times:.2f}秒")

def process_single_item(item, idx):
    """處理單一項目，立即釋放記憶體"""
    if 'full_graph' not in item or item['full_graph'] is None:
        return None
    
    graph_data = item['full_graph']
    is_vul = 'cwe_id' in item and item['cwe_id'] is not None and item['cwe_id'] != '' and item['cwe_id'] != 'normal'
    
    try:
        # 處理邏輯
        if is_vul:
            target_lines = item['buggy_lines'].keys()
        
        if target_lines:
            sliced_data_buggy = slice_graph(graph_data, mode=SLICE_MODE, target_lines=target_lines)
            item['sliced_graph_buggy'] = sliced_data_buggy
        else:
            logging.warning(f"⚠️ 第 {idx} 筆資料無 target_lines")

        # 🔍 可選：也裁 CALL/IDENTIFIER（V2 模式）
        # sliced_data_node_type = slice_graph(graph_data, mode=SLICE_MODE)
        # item['sliced_graph_node_type'] = sliced_data_node_type

        
        # 立即清理原圖以節省記憶體
        # del item['full_graph']
        
        return item
        
    except Exception as e:
        logging.error(f"處理項目 {idx} 失敗: {e}")
        return None

def merge_temp_files(temp_dir, output_path):
    """合併臨時檔案"""
    all_data = []
    
    temp_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.pt')])
    
    for temp_file in tqdm(temp_files, desc="合併檔案"):
        temp_path = os.path.join(temp_dir, temp_file)
        batch_data = torch.load(temp_path)
        all_data.extend(batch_data)
        
        # 立即刪除已讀取的臨時檔案
        os.remove(temp_path)
    
    # 保存最終結果
    torch.save(all_data, output_path)
    logging.info(f"✅ 合併完成，總共 {len(all_data)} 筆資料")



def main():
    for pt, out in [(EXAMPLE_PT, EXAMPLE_OUT_PT)]:
    # for pt, out in [(TRAIN_PT, TRAIN_OUT), (EVAL_PT, EVAL_OUT), (TEST_PT, TEST_OUT)]:
        if os.path.exists(pt):
            process_pt_dataset_memory_safe(pt, out, batch_size=50)
        else:
            logging.warning(f"❌ File not found: {pt}")

if __name__ == "__main__":
    main()