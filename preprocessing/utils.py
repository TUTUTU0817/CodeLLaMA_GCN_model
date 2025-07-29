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

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CWE_CLASSES = [
    # "normal",   # 正常程式碼
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

def convert_jsonl_to_pt(jsonl_path):
    """將 result.jsonl 轉為 PyTorch 的 .pt 格式"""
    import torch
    import json

    pt_path = jsonl_path.replace(".jsonl", ".pt")
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("label") != 1:
                continue
            apk_name = item["apk_name"]
            file_path = item["file"].replace('/', '_').replace(".java", "")
            full_name = item["fullName"].replace("/", "_").replace(":", "_").replace("(", "_").replace(")", "_").replace(",", "_").replace(" ", "_")
            line_number = item["lineNumberStart"]
            key = f"{apk_name}_{file_path}_{full_name}_{line_number}"
            item["key"] = key
            data.append(item)

    torch.save(data, pt_path)
    return pt_path