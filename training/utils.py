import random
import numpy as np
import torch
import json
from networkx.drawing.nx_pydot import read_dot
from torch_geometric.data import Data
import os
import pickle
from sklearn.metrics import classification_report
from datasets import load_dataset
from transformers import  AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CWE_CLASSES = [
    # "normal",
    "CWE-89",   # SQL Injection
    "CWE-200",  # Information Exposure
    "CWE-250",  # Execution with Unnecessary Privileges
    "CWE-276",  # Incorrect Default Permissions
    "CWE-295",  # Improper Certificate Validation
    "CWE-312",  # Cleartext Storage of Sensitive Information
    "CWE-327",  # Use of a Broken or Risky Cryptographic Algorithm
    "CWE-649",  # Reliance on Obfuscation or Encryption Without Integrity
    "CWE-749",  # Exposed Dangerous Method or Function
]

LABEL_TO_CWE = {i: cwe for i, cwe in enumerate(CWE_CLASSES)}
CWE_TO_LABEL = {cwe: i for i, cwe in enumerate(CWE_CLASSES)}


node_type_dict = {
    'ANNOTATION_LITERAL': 0, 
    'ANNOTATION_PARAMETER': 1, 
    'ANNOTATION_PARAMETER_ASSIGN': 2, 
    'ARRAY_INITIALIZER': 3, 
    'BLOCK': 4, 
    'CONTROL_STRUCTURE': 5,
    'FIELD_IDENTIFIER': 6, 
    'IDENTIFIER': 7,
    'JUMP_TARGET': 8,
    'LITERAL': 9, 
    'LOCAL': 10,  
    'METHOD': 11, 
    'METHOD_RETURN': 12, 
    'MODIFIER': 13,
    'PARAM': 14,
    'RETURN': 15,
    'TYPE_REF': 16,
    'UNKNOWN': 17,
}

edge_type_dict = {
    "AST": 0,
    "CFG": 1,
    "DDG": 2,
    "CDG": 3,
}

# # 讀取資料
# def load_json_dataset(json_file):
#     """ 讀取 JSON 訓練集 """
#     with open(json_file, 'r') as f:
#         dataset = json.load(f)
#     return dataset

# # 圖的前處理
# def get_code_embedding(llm_model, tokenizer, code_snippet):
#     """ 取得 LLM 的 code embedding（mean pooling）"""
#     tokens = tokenizer(code_snippet, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
#     with torch.no_grad():
#         embedding = llm_model.base_model(**tokens).last_hidden_state.mean(dim=1)
#     return embedding.squeeze(0)


# def create_pyg_graph_from_dot(llm_model, tokenizer, dot_file):
#     """ 讀取 DOT 圖並轉換為 PyG 圖 """
#     nx_graph = read_dot(dot_file)
#     node_map = {node: i for i, node in enumerate(nx_graph.nodes())}
#     edge_index = torch.tensor([
#         [node_map[src], node_map[dst]] for src, dst in nx_graph.edges()
#     ]).T

#     node_features = []
#     for node in nx_graph.nodes():
#         code = nx_graph.nodes[node].get("code", "")
#         emb = get_code_embedding(llm_model, tokenizer, code).to("cpu")
#         node_features.append(emb)
#     x = torch.stack(node_features)
#     return Data(x=x, edge_index=edge_index)

# def get_pyg_graph(model, tokenizer, dot_file, cache_dir="/home/tu/exp/Training/GNN/cache"):
#     """ 
#     讀取 JSON 訓練集，從 DOT 檔案創建 PyG 圖，並返回資料集。
    
#     參數:
#     - model: 用來生成程式碼嵌入的模型（Code LLaMA）。
#     - tokenizer: 用來處理程式碼的分詞器（Code LLaMA）。
#     - dot_file: 包含圖結構的 dot 文件。
#     - label_map: CWE 類別的對應映射。
#     - cache_dir: 快取資料夾，用來儲存已處理的圖。
    
#     返回:
#     - pyg_graph: 圖。
#     """
#     os.makedirs(cache_dir, exist_ok=True)  # 確保快取資料夾存在

#     cache_file = os.path.join(cache_dir, os.path.basename(dot_file) + ".pkl")
    
#     # 如果有快取，就直接讀取
#     if os.path.exists(cache_file):
#         with open(cache_file, "rb") as f:
#             pyg_graph = pickle.load(f) 
#     else:
#         pyg_graph = create_pyg_graph_from_dot(model, tokenizer, dot_file)
#         with open(cache_file, "wb") as f:
#             pickle.dump(pyg_graph, f)
#     return pyg_graph


def load_model(num_labels, model_name="codellama/CodeLlama-7b-Instruct-hf"):
    """
    載入 Code LLaMA 並進行 4-bit 量化，設定 QLoRA。
    
    :param model_name: Hugging Face 上的模型名稱
    :param num_labels: 漏洞類型數量
    :return: 量化後的 QLoRA 模型 & tokenizer
    """

    # ✅ 使用 bitsandbytes 來進行 4-bit 量化，確保 QLoRA 設定正確
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # 進一步壓縮權重，減少記憶體使用
        bnb_4bit_quant_type="nf4",  # Normal Float 4，適合 QLoRA 訓練
        bnb_4bit_compute_dtype=torch.bfloat16  # 避免 FP16 計算時出現 NaN
    )


    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=quantization_config,
        # device_map="auto"
        device_map={"": 0}
    )

    # model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def load_fine_tuned_model(num_labels, model_name, checkpoint_path):
    """
    載入微調後的模型
    :param model_name: 模型名稱
    :return: 載入的模型與 tokenizer
    """

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # ✅ 4-bit 量化（若用 8-bit，改為 load_in_8bit=True）
        bnb_4bit_compute_dtype=torch.float16,  
        bnb_4bit_use_double_quant=True,  
        bnb_4bit_quant_type="nf4"  
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        quantization_config=quantization_config,
        device_map={"": 0}  
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    # print(model.peft_config)
    return model, tokenizer


def load_data(train_file, test_file, eval_file):
    """
    載入 JSON 格式的 function 資料，並轉換為 Hugging Face dataset 格式。

    :param train_file: 訓練集 JSON 檔案
    :param test_file: 測試集 JSON 檔案
    :return: 訓練集 & 測試集
    """

    def preprocess_json(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        for item in data:
            if isinstance(item["buggy_lines"], dict):
                item["buggy_lines"] = "\n".join([f"Buggy Line {line}: {code}" for line, code in item["buggy_lines"].items()])
        
        # 把處理後的 JSON 存回原檔案
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    # 預處理並載入資料
    preprocess_json(train_file)
    preprocess_json(test_file)
    preprocess_json(eval_file)
    dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file, "test": test_file})
    return dataset["train"],  dataset["eval"], dataset["test"]


def setup_tokenizer(model, tokenizer):
    """
    設定 tokenizer 以適應 function 級別的訓練資料。
    """
    tokenizer.pad_token = tokenizer.eos_token  # 🔹 把 `pad_token` 設定成 `eos_token`
    tokenizer.pad_token_id = tokenizer.eos_token_id  # 🔹 確保 `pad_token_id` 設定正確
    tokenizer.padding_side = "left"  # Code Llama 預設左側 `padding`
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer



def build_prompt(code, buggy_lines):
    """
    <s>
        [INST]
            <<SYS>>{{ system_prompt }}<</SYS>>
        {{ user_msg_1 }}
        [/INST]
        {{ model_amswer_1 }}
    </s>
    <s>
        [INST]
        {{ user_msg_2 }}
        [/INST]
        {{ model_amswer_2 }}
    </s>
    """
    
    system_prompt = f"""You are a cybersecurity expert. 
                        Based on the given code and buggy line, Answer with only one CWE-ID:\n.
                        Do not explain or provide details."""
    user_msg = f""" [function_code]:\n{code}\n
                    [known buggy line]:\n{buggy_lines}\n
                    [Response]:"""
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_msg} [/INST]</s>" 

def build_prompt_B(code):
    """
    <s>
        [INST]
            <<SYS>>{{ system_prompt }}<</SYS>>
        {{ user_msg_1 }}
        [/INST]
        {{ model_amswer_1 }}
    </s>
    <s>
        [INST]
        {{ user_msg_2 }}
        [/INST]
        {{ model_amswer_2 }}
    </s>
    """
    
    system_prompt = f"""You are a cybersecurity expert. 
                        Based on the given code and buggy line, Answer with only one CWE-ID:\n.
                        Do not explain or provide details."""
    user_msg = f""" [function_code]:\n{code}\n
                    [Response]:"""
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_msg} [/INST]</s>" 

# def tokenize(tokenizer, sample):
#     system_prompt = "You are a cybersecurity expert. Identify the CWE-ID of the following buggy code.Answer with only one CWE-ID (e.g., CWE-200). Do not explain or provide details."
#     user_msg = f"[function_code]:\n{sample['code']}\n[known buggy line]:\n{sample['buggy_line']}\n[Response]:"
#     target = sample["cwe_id"]
#     full_prompt = build_prompt(system_prompt, user_msg, target)

#     tokenized = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=1024, return_tensors="pt")
#     return {
#         "input_ids": tokenized["input_ids"].squeeze(0),
#         "attention_mask": tokenized["attention_mask"].squeeze(0)
#     }

# def tokenize(tokenizer,sample):
#     """
#     - 將 function 轉換為 Token，並將 function 本身當作 labels 進行自監督學習。
#     """

#     prompt = build_prompt(sample["code"], sample["buggy_lines"])
#     tokenized = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length")

#     # input_ids = tokenized["input_ids"].squeeze(0)
#     # attention_mask = tokenized["attention_mask"].squeeze(0)
    
#     # 將 cwe_id 轉為整數標籤
#     # labels = torch.tensor(CWE_TO_LABEL[sample["cwe_id"]], dtype=torch.long)
#     tokenized["labels"] = CWE_TO_LABEL[sample["cwe_id"]]
#     # labels = input_ids.clone() # Causal LM 要求 labels 與 input_ids 一樣
#     # return {
#     #     "input_ids": input_ids, 
#     #     "attention_mask": attention_mask,
#     #     "labels": labels,
#     # }
#     return tokenized

def get_unique_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    counter = 2
    while True:
        new_path = f"{base_path}_{counter}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1
        
def get_unique_filename(output_path, base_name):
    counter = 1
    while True:
        file_name = f"{base_name}_batch_{counter}.pkl"
        file_path = os.path.join(output_path, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1

def save_result(results, report, output_dir, training_time, config):
    """
    儲存預測結果與分類報告到指定的目錄。
    :param results: 預測結果
    :param report: 分類報告
    :param output_dir: 儲存的目錄
    :param training_time: 訓練時間
    :param config: 模型配置
    """

    json_path = get_unique_path(os.path.join(output_dir, "best_fusion_prediction_results")) + ".json"
    with open(json_path, "w") as jf:
        json.dump(results, jf, indent=2)
    print(f"✅ 預測結果已儲存至: {json_path}")
    
    txt_path = get_unique_path(os.path.join(output_dir, "best_fusion_classification_report")) + ".txt"
    with open(txt_path, "w") as tf:
        tf.write(report)
        tf.write(f"\n\n🕒 Training time: {round(training_time, 2)} seconds\n")
        tf.write(f"\n\n📝 Model configuration:\n {config}")
    print(f"✅ 分類報告已儲存至: {txt_path}")


# def save_result(results, result_output_dir, report_output_dir, training_time, config):


#     # 檢查並生成唯一的檔案名稱
#     def get_unique_file_name(file_path):
#         if os.path.exists(file_path):
#             base_name, ext = os.path.splitext(file_path)
#             counter = 2
#             while os.path.exists(f"{base_name}_{counter}{ext}"):
#                 counter += 1
#             return f"{base_name}_{counter}{ext}"
#         return file_path
#     y_true = [res["y_true"] for res in results]
#     y_pred = [res["y_pred"] for res in results]
    
#     report = classification_report(
#         y_true, 
#         y_pred, 
#         target_names=LABEL_TO_CWE.values(), 
#         digits=4,
#         labels=list(LABEL_TO_CWE.keys()),  # 指定要計算的類別
#         zero_division=0  # 若某類別沒出現，也不報錯
# )

#     result_output_dir = get_unique_file_name(result_output_dir)
#     report_output_dir = get_unique_file_name(report_output_dir)

#     with open(result_output_dir, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)

#     with open(report_output_dir, "w", encoding="utf-8") as f:
#         f.write(report)  # 接寫入 txt 檔案
#         f.write(f"\n\n🕒 Training time: {round(training_time, 2)} seconds\n")
#         f.write(f"\n\n📝 Model configuration:\n {config}")
#     return report


class JsonGraphDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, cwe2idx):
        with open(json_file) as f:
            entries = json.load(f)

        self.entries = [e for e in entries if e.get("graph") and os.path.exists(e["graph"])]
        self.cwe2idx = cwe2idx

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        data = torch.load(entry["graph"], map_location='cpu')
        
        cwe_id = entry["cwe_id"]
        label_idx = self.cwe2idx.get(cwe_id, -1)  # 預設 -1 代表未知
        assert label_idx >= 0, f"Unknown CWE-ID: {cwe_id}"

        data.y = torch.tensor(label_idx, dtype=torch.long)
        data.file_name = entry["fullName"]  # optional
        return data
    
def save_result(results, report, output_dir, model_name):

    json_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(json_path, "w") as jf:
        json.dump(results, jf, indent=2)
    print(f"✅ 預測結果已儲存至: {json_path}")
    
    txt_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
    with open(txt_path, "w") as tf:
        tf.write(report)
    print(f"✅ 分類報告已儲存至: {txt_path}")


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 時也固定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(data, labels, random_statem, test_size=0.1):
    """
    根據標籤進行分層抽樣，將資料分為訓練集和測試集。
    """
    from sklearn.model_selection import train_test_split

    # 使用 stratify 參數進行分層抽樣
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_statem, stratify=labels
    )
    # 再從剩餘的訓練集中切出驗證集
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        train_data, train_labels, test_size=test_size / (1 - test_size), random_state=random_statem, stratify=train_labels)
    
    return  train_data, train_labels, eval_data, eval_labels, test_data, test_labels


