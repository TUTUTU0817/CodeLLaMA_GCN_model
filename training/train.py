import argparse
import os
import pickle
import random
import pandas as pd
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as GraphBatch
import torch
from utils import CWE_CLASSES, save_result, device, load_model, load_fine_tuned_model, setup_tokenizer, set_seed, stratified_split, CWE_TO_LABEL
import yaml
import time
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import sys
sys.path.append('/home/tu/exp')  # 讓 Training 被找到
from Training.GNN.AZOO.model import GGNN, GATModel


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config


class LoadMultiDataset(Dataset):
    """
    自定義 Dataset 類別，用於載入二進制序列化的圖數據。
    """
    def __init__(self, data):
        self.data = data
        self.valid_indices = []
        for idx, item in enumerate(data):
            full_graph = item.get("full_graph")
            if (full_graph is not None and 
                hasattr(full_graph, 'x') and 
                full_graph.x is not None and 
                full_graph.x.size(0) > 0):
                # if 'sliced_graph_node_type' in item and 'cls_embed' in item and 'ft_cls_embed' in item:
                if 'sliced_graph_buggy' in item and 'cls_embed' in item and 'ft_cls_embed' in item:
                    self.valid_indices.append(idx)
        
        self.data = data
        print(f"Filtered dataset: {len(self.valid_indices)} valid items out of {len(data)} total items")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]  # 直接從 .pt 文件中獲取數據
        item = self.data[actual_idx]

        full_graph = item["full_graph"]

        
        # 將 cwe_id 字符串轉換為數字標籤
        cwe_id = item["cwe_id"]
        label = CWE_TO_LABEL[cwe_id]
        sliced_graph_buggy = item["sliced_graph_buggy"]
        if sliced_graph_buggy is not None and (not hasattr(sliced_graph_buggy, 'x') or sliced_graph_buggy.x is None or sliced_graph_buggy.x.size(0) == 0):
            sliced_graph_buggy = None

        # sliced_graph_node_type = item["sliced_graph_node_type"]
        # if sliced_graph_node_type is not None and (not hasattr(sliced_graph_node_type, 'x') or sliced_graph_node_type.x is None or sliced_graph_node_type.x.size(0) == 0):
        #     sliced_graph_node_type = None

        return {
            "key": item["key"],
            "full_graph": full_graph,
            "sliced_graph_buggy": sliced_graph_buggy,
            # "sliced_graph_node_type": sliced_graph_node_type,
            "cls_embed": item["cls_embed"],
            "ft_cls_embed": item["ft_cls_embed"],
            "label": label,
        }


# def fusion_collate_fn_B(samples):
#     """
#     自定義 collate 函數，用於處理批次數據。
#     將樣本列表轉換為批次格式，並處理圖數據。
#     """
#     labels = torch.tensor([s['label'] for s in samples], dtype=torch.long)
    
#     # 處理圖數據
#     graphs = []
#     for s in samples:
#         if s['full_graph'] is not None:
#             if hasattr(s['full_graph'], 'x') and s['full_graph'].x is not None and s['full_graph'].x.size(0) > 0:
#                 graphs.append(s['full_graph'])

#     graph_batch = GraphBatch.from_data_list(graphs) if graphs else None

#     result = {
#         "key": [s['key'] for s in samples],
#         "labels": labels,
#     }

#     if graph_batch is not None:
#         result["graph"] = graph_batch

#     return result

def fusion_collate_fn(samples):
    labels = torch.tensor([s['label'] for s in samples], dtype=torch.long)
    graphs = [s['full_graph'] for s in samples if s['full_graph'] is not None]
    sliced_graph_buggy = [s['sliced_graph_buggy'] for s in samples if s['sliced_graph_buggy'] is not None]
    # sliced_graph_node_type = [s['sliced_graph_node_type'] for s in samples if s['sliced_graph_node_type'] is not None]
    graph_batch = GraphBatch.from_data_list(graphs) if graphs else None
    sliced_graph_buggy = GraphBatch.from_data_list(sliced_graph_buggy) if sliced_graph_buggy else None
    # sliced_graph_node_type = GraphBatch.from_data_list(sliced_graph_node_type) if sliced_graph_node_type else None
    return {
        "key": [s['key'] for s in samples],
        "labels": labels,
        "full_graph": graph_batch,
        "cls_embed": torch.stack([s['cls_embed'] for s in samples]),
        "ft_cls_embed": torch.stack([s['ft_cls_embed'] for s in samples]),
        "sliced_graph_buggy": sliced_graph_buggy,
        # "sliced_graph_node_type": sliced_graph_node_type,
    }

# class ClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks.
#     - 參數:
#         - config: LLM 模型配置參數
#         - extra_dim: GNN 模型輸出維度
#     """

#     def __init__(self, config, input_dim, num_classes=9, dropout=0.1):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, config.hidden_size)
#         self.act = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(config.hidden_size, num_classes)

#     def forward(self, features):
#     # def forward(self, features, ggnn_embed, **kwargs):
#         # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         # if ggnn_embed is not None:
#         #     x = torch.cat((x, ggnn_embed), dim=1)
#         x = features
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks with multiple layers."""

    def __init__(self, config, input_dim, num_classes=9, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        hidden_dim = config.hidden_size

        # 第一層
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # 中間層
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 最終輸出層
        self.output_layer = nn.Linear(hidden_dim, num_classes)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        x = features
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)  # ReLU
            x = self.dropout(x)  # Dropout
        x = self.output_layer(x)  # 最終輸出層
        return x

class GCNModel(nn.Module):
    """ GCN 模型 (4層) """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, out_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, out_dim)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # x = self.conv1(x, edge_index).relu()
        # x = F.dropout(x, p=0.3, training=self.training)  # Dropout
        # x = self.conv2(x, edge_index)
        # x = global_mean_pool(xㄋ, batch)
        # return F.log_softmax(x, dim=1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm2(x)
        # x = self.batch_norm2(x)

        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv3(x, edge_index).relu()
        x = self.batch_norm3(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv4(x, edge_index)
        x = global_mean_pool(x, batch)  # 聚合節點特徵

        # return F.log_softmax(x, dim=1)
        return x

def calculate_class_weights(labels, num_classes):
    """計算類別權重來處理不平衡數據"""
    from collections import Counter
    
    # 統計每個類別的樣本數
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # 計算權重：總樣本數 / (類別數 * 該類別樣本數)
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
            weights.append(weight)
        else:
            weights.append(1.0)  # 如果某類別沒有樣本
    
    return torch.tensor(weights, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss


        # 添加類別權重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss



        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



class LLMGNNModel_fusion(nn.Module):
    """
    融合 LLM 和 GNN 的模型。
    根據 mode 參數選擇使用 LLM 嵌入、GNN 嵌入或兩者的拼接。
    """
    def __init__(self, gnn_encoder, config, tokenizer, mode, num_classes, num_layers=2):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.tokenizer = tokenizer
        self.mode = mode

        if self.mode == "llm":
            input_dim = config.hidden_size
        elif self.mode == "gnn":
            input_dim = gnn_encoder.out_dim
        elif self.mode == "concat":
            input_dim = config.hidden_size + gnn_encoder.out_dim
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        self.classifier = ClassificationHead(config=config, input_dim=input_dim, num_classes=num_classes,  num_layers=num_layers)

    def forward(self, labels=None, graphs=None, CLS_token=None):
        if CLS_token is not None:
            CLS_token = CLS_token.to(dtype=self.classifier.layers[0].weight.dtype)

        if self.mode in ["gnn", "concat"] and graphs is not None:
            gnn_embedding = self.gnn_encoder(
                x=graphs.x,
                edge_index=graphs.edge_index,
                edge_attr=None,
                batch=graphs.batch
            )
            gnn_embedding = gnn_embedding.to(dtype=self.classifier.layers[0].weight.dtype)

        # 根據模式選擇特徵
        if self.mode == "llm":
            features = CLS_token
        elif self.mode == "gnn":
            features = gnn_embedding
        elif self.mode == "concat":
            features = torch.cat([CLS_token, gnn_embedding], dim=-1)

        logits = self.classifier(features)
        # loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))  # 使用類別權重
        loss_fct = FocalLoss(gamma=2)  # 使用 Focal Loss
        loss = loss_fct(logits, labels)
        prob = torch.softmax(logits, dim=-1)

        return loss, prob


# class EarlyStopping:
#     def __init__(self, patience=7, delta=0.001, verbose=True):  # 降低 delta
#         self.patience = patience
#         self.delta = delta
#         self.best_score = None
#         self.counter = 0
#         self.early_stop = False
#         self.verbose = verbose

#     def __call__(self, val_score):  # 改為 val_score，直接使用 F1
#         score = val_score  # 直接使用 F1-score
        
#         if self.best_score is None:
#             self.best_score = score
#             if self.verbose:
#                 print(f"初始最佳 F1-score: {score:.4f}")
#         elif score > self.best_score + self.delta:  # F1 越大越好
#             self.best_score = score
#             self.counter = 0
#             if self.verbose:
#                 print(f"新的最佳 F1-score: {score:.4f}")
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"F1-score 沒有改善 ({self.counter}/{self.patience})")
            
#             if self.counter >= self.patience:
#                 self.early_stop = True
#                 if self.verbose:
#                     print(f"早停觸發！最佳 F1-score: {self.best_score:.4f}")

def train_model(config, fusion_model, train_loader, eval_loader, model_name, output_dir):
    
    log_file = os.path.join(output_dir, "train_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_acc,precision,recall,f1_score\n")

    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=float(config["lr"]))
    
    best_val_f1 = 0.0

    train_start = time.time()
    # early_stopping = EarlyStopping(patience=7)
    for epoch in range(config["epochs"]):
        epoch_start = time.time()  # ⏱️ 開始時間

        fusion_model.train()
        total_loss = 0.0

        # all_preds = []
        # all_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            
            if config["llm_type"] == "nft":
                cls_embed = batch[f"cls_embed"].to(device)
            elif config["llm_type"] == "ft":
                cls_embed = batch[f"ft_cls_embed"].to(device)

            # gnn_embed = batch[f"{config['graph_type']}_{config['gnn_model']}_embed"].to(device)

            if config['graph_type'] == "full":
                # 處理圖數據
                graph = batch["full_graph"].to(device)  # 將 graph 傳入模型
            elif config['graph_type'] == "sliced":
                # 處理圖數據
                graph = batch["sliced_graph_buggy"].to(device)
                # graph = batch["sliced_graph_node_type"].to(device)

            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, probs = fusion_model(labels, graph, cls_embed)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # preds = probs.argmax(dim=1)
            # all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())


        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch}: Train Loss = {total_loss:.4f}")
        # ⏱️ 結束時間與紀錄
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"⏰ Epoch {epoch} 訓練時間：{epoch_time:.2f} 秒")
        

        # Validation
        _, val_report, _ = evaluate_model(config, fusion_model, eval_loader)
        
        val_acc = val_report["accuracy"]
        val_precision = val_report["macro avg"]["precision"]
        val_recall = val_report["macro avg"]["recall"]
        val_f1 = val_report["macro avg"]["f1-score"]
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(output_dir, f"best_fusion_model_{model_name}") + ".pt"
            torch.save(fusion_model.state_dict(), save_path)
            print(f"🌟 新最佳模型！Val f1-score = {val_f1:.4f}")

        # 早停檢查 - 直接使用 F1-score
        # early_stopping(val_f1)  
        # if early_stopping.early_stop:
        #     print(f"⏹️ 早停觸發於 Epoch {epoch}，載入最佳模型...")
        #     fusion_model.load_state_dict(torch.load(save_path))
        #     break

        # Logging
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_loss:.4f},{val_acc:.4f},{val_precision:.4f},{val_recall:.4f},{val_f1:.4f}\n")
            
        
    train_end = time.time()
    total_train_time = train_end - train_start
    print(f"🚀 總訓練時間：{total_train_time:.2f} 秒")
    
    summary_path = os.path.join(output_dir, "training_summary.yaml")
    summary = dict(config)  # shallow copy
    summary["total_train_time_sec"] = total_train_time
    
    with open(summary_path, "w") as f:
        yaml.dump(summary, f)

    print(f"📝 訓練摘要已儲存至：{summary_path}")
    
def evaluate_model(config, fusion_model, loader):

    fusion_model.eval()
    all_preds, all_labels = [], []
    results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            if config["llm_type"] == "nft":
                cls_embed = batch[f"cls_embed"].to(device)
            elif config["llm_type"] == "ft":
                cls_embed = batch[f"ft_cls_embed"].to(device)
                
            # gnn_embed = batch[f"{config['graph_type']}_{config['gnn_model']}_embed"].to(device)
            
            if config['graph_type'] == "full":
                # 處理圖數據
                graph = batch["full_graph"].to(device)  # 將 graph 傳入模型
            elif config['graph_type'] == "sliced":
                # 處理圖數據
                graph = batch["sliced_graph_buggy"].to(device)
                # graph = batch["sliced_graph_node_type"].to(device)
                # graph = batch["full_graph"].to(device)  # 將 graph 傳入模型

            labels = batch["labels"].to(device)

            _, probs = fusion_model(labels, graph, cls_embed)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for i, (true, pred) in enumerate(zip(batch["labels"].cpu().tolist(), preds.cpu().tolist())):
                results.append({
                    "filename": batch["key"][i] if "key" in batch else f"data_{i}",
                    "true_label": int(true),
                    "predicted_label": int(pred)
                })
                    
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    text_report = classification_report(all_labels, all_preds, digits=4)    
    # print("\n📄 Classification Report:")
    # print(text_report)

    return results, report, text_report

def main():
    
    # seed=77
    # set_seed(seed)
    config = load_config()
    
    
    batch_size = config.get("batch_size", 32)
    fusion_type = config['fusion_type']
    
    if config["task"] == "multi":
        # all_data = torch.load(config["all_data_file"])
        # labels = [CWE_TO_LABEL[d["cwe_id"]] for d in all_data]
        num_classes = len(CWE_CLASSES)
        llm = "llm_path"
    #     # 計算類別權重
        # class_weights = calculate_class_weights(labels, num_classes)

    elif config["task"] == "binary":
    #     all_data = torch.load(config["all_data_file_binary"])  # 載入 .pt 文件
    #     labels = [d["label"] for d in all_data]
        num_classes = 2
        llm = "llm_path_binary"

    # else:
    #     raise ValueError(f"Unsupported 'task': {config['task']}. Use 'multi' or 'binary'")


    print("🔍 載入 LLM...")
    llm_type = config["llm_type"]
    if llm_type == "ft":
        print("載入 fine-tuned LLM ")
        model, tokenizer = load_fine_tuned_model(num_classes, "codellama/CodeLlama-7b-Instruct-hf", config[llm])
        # print(model.config.hidden_size)
    elif llm_type =="nft":
        print("載入 Original LLM ")
        model, tokenizer = load_model(num_classes, "codellama/CodeLlama-7b-Instruct-hf")
    else:
        raise ValueError(f"Unsupported 'llm_type': {config['llm_type']}. Use 'ft' or 'nft' ")
    
    print("⚙️ 設定 tokenizer ...")
    tokenizer = setup_tokenizer(model, tokenizer)
        
    if fusion_type in ["gnn", "concat"]:
        gnn_model_name = config["gnn_model"]
        print(f"🧠 載入 {gnn_model_name} 模型參數...")
        if gnn_model_name == "ggnn":
            gnn_model = GGNN(
                in_dim=4160,  # LLM + type embedding
                hidden_dim=128,
                edge_dim=16,
                num_steps=5,
                out_dim=128
            ).to(device)
        elif gnn_model_name == "gcn":
            gnn_model = GCNModel(
                in_dim=4160, # LLM + type_embedding 維度
                hidden_dim=config["gnn_out_dim"],  # FIXME
                out_dim=config["gnn_out_dim"]   # FIXME
            ).to(device)        
        elif gnn_model_name == "gat":
            gnn_model = GATModel(
                in_dim=4160, # LLM + type_embedding 維度
                hidden_dim=128, 
                out_dim=128,   # 要分類的 embedding 維度
                num_heads=4,  # 多頭注意力的頭數
            ).to(device)
        else:
            raise ValueError(f"Unsupported 'gnn_model': {config['gnn_model']}. Use 'ggnn' or 'gcn' or 'gat' ")
        
        graph_type = config["graph_type"]
        if graph_type not in ["full", "sliced"]:
            raise ValueError(f"Unsupported 'graph_type': {config['graph_type']}. Use 'full' or 'sliced'")
        # print(f"🧠 載入 {graph_type}_graph ...")    
        # gnn_ckpt = torch.load(config[f"{graph_type}_{gnn_model_name}_path"], map_location=device, weights_only=True)
        # gnn_model.load_state_dict(gnn_ckpt["gnn"])
    else:
        gnn_model = None

    print("🧠 構建融合模型...")
    
    print(f"🧠 Building model in {fusion_type} ...")
    if fusion_type == "llm":
        model_name = f"{fusion_type}_{llm_type}"
    elif fusion_type == "gnn":
        model_name = f"{fusion_type}_{graph_type}_{gnn_model_name}"
    elif fusion_type == "concat":
        model_name = f"{fusion_type}_{llm_type}_llm_{graph_type}_{gnn_model_name}"
    else:        
        raise ValueError(f"Unsupported 'fusion_type': {config['fusion_type']}. Use 'llm' or 'gnn' or 'concat'")
    model_name = model_name
    
    # fusion_model = LLMGNNModel_fusion(
    #     gnn_encoder=gnn_model,
    #     config=model.config,
    #     tokenizer=tokenizer,
    #     mode=fusion_type,  # "llm", "gnn", "concat"
    # ).to(device)
    # llm_model = LLMModel(model, tokenizer).to(device)


    print("loading Dataset....")
    
    
    best_seed = None
    best_f1 = -1
    all_results = []

    # train_data= torch.load("/home/tu/exp/EXP_final/Dataset/multi/train_tmp_data.pt")
    # val_data= torch.load("/home/tu/exp/EXP_final/Dataset/multi/eval_tmp_data.pt")
    # test_data= torch.load("/home/tu/exp/EXP_final/Dataset/multi/test_tmp_data.pt")
    train_data = torch.load("/home/tu/exp/EXP_final/Dataset/multi/vul_train.pt")
    val_data = torch.load("/home/tu/exp/EXP_final/Dataset/multi/vul_eval.pt")
    test_data = torch.load("/home/tu/exp/EXP_final/Dataset/multi/vul_test.pt")
    train_data = LoadMultiDataset(train_data)
    val_data = LoadMultiDataset(val_data)
    test_data = LoadMultiDataset(test_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=fusion_collate_fn)
    eval_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=fusion_collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=fusion_collate_fn)
    # for num_layers in range(2, 3):
    num_layers = 3
    print(f"\n🔧 開始測試 {num_layers} 層分類器...")

    for i in range(config["num_trails"]):
        
        # rs = random.randint(0, 10000)
        # rs = i + 1  # 使用 1 到 100 的隨機種子
        rs = int(817)
        set_seed(rs)  # 設定隨機種子

        # rs = 28

        # train_data, train_labels, val_data, val_labels, test_data, test_labels = stratified_split(
        #     all_data, 
        #     labels, 
        #     rs,
        #     test_size=0.1)
        
        # train_data = LoadMultiDataset(train_data)
        # val_data = LoadMultiDataset(val_data)
        # test_data = LoadMultiDataset(test_data)

        # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=fusion_collate_fn)
        # eval_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=fusion_collate_fn)
        # test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=fusion_collate_fn)

        # # 測試一個小批次
        # try:
        #     small_batch = [train_data[i] for i in range(min(4, len(train_data)))]
        #     test_result = fusion_collate_fn(small_batch)
        #     print("✅ Collate function test passed")
        # except Exception as e:
        #     print(f"❌ Collate function test failed: {e}")



        print("Dataset loaded successfully with Random Seed:", rs)

        fusion_model = LLMGNNModel_fusion(
            gnn_encoder=gnn_model,
            config=model.config,
            tokenizer=tokenizer,
            mode=fusion_type,  # "llm", "gnn", "concat"
            num_classes=num_classes,
            num_layers=num_layers,
            # class_weights=class_weights
        ).to(device)


        # rs_model_name=f"{rs}_{model_name}_{config['task']}_num_layer{num_layers}_{i}"
        rs_model_name=f"{rs}_{model_name}_{config['task']}_{config['gnn_out_dim']}_n_{i}"

        output_dir = os.path.join(config['output_dir'], rs_model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        mode = config["mode"]
        if mode == "train":
            print(" 🔧 訓練模型...")
            train_model(config=config, 
                        fusion_model=fusion_model, 
                        train_loader=train_loader,
                        eval_loader=eval_loader, 
                        model_name=rs_model_name, 
                        output_dir=output_dir)
        elif mode == "test":
            print("🔍 評估模型...")
            output_dir = "/home/tu/exp/EXP_final/result/EXP4/817_concat_ft_llm_sliced_gcn_multi_num_layer3_8/test"
            # best_model_path = os.path.join(output_dir, f"best_fusion_model_{rs_model_name}.pt")
            best_model_path = "/home/tu/exp/EXP_final/result/EXP4/817_concat_ft_llm_sliced_gcn_multi_num_layer3_8/best_fusion_model_817_concat_ft_llm_sliced_gcn_multi_num_layer3_8.pt"
            fusion_model.load_state_dict(torch.load(best_model_path, map_location=device))
            fusion_model.eval()
            start_time = time.time()
            results, report, text_report = evaluate_model(config, fusion_model, test_loader)
            end_time = time.time()
            print(f"⏱️ 評估時間：{end_time - start_time:.2f} 秒")

            save_result(results=results, 
                        report=text_report, 
                        output_dir=output_dir, 
                        model_name=rs_model_name + "_best")
            # 取 macro avg F1-score
            f1 = report["macro avg"]["f1-score"]
            all_results.append((rs, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_seed = rs
        elif mode == "auto":
            print("🔍 自動模式：訓練並評估模型...")
            train_model(config=config, 
                        fusion_model=fusion_model, 
                        train_loader=train_loader,
                        eval_loader=eval_loader, 
                        model_name=rs_model_name, 
                        output_dir=output_dir)
            
            best_model_path = os.path.join(output_dir, f"best_fusion_model_{rs_model_name}.pt")
            fusion_model.load_state_dict(torch.load(best_model_path, map_location=device))
            fusion_model.eval()
            results, _, text_report = evaluate_model(config, fusion_model, test_loader)
            save_result(results=results, 
                        report=text_report, 
                        output_dir=output_dir, 
                        model_name=rs_model_name + "_best")
        else:
            raise ValueError(f"Unsupported 'mode': {config['mode']}. Use 'train' or 'test'")
            

    print("==== 所有 seed 的 F1-score ====")
    for rs, f1 in all_results:
        print(f"Seed {rs}: F1-score = {f1:.4f}")
    print(f"\n🏆 最佳 seed: {best_seed}，F1-score = {best_f1:.4f}")




    
if __name__ == "__main__":
    main()