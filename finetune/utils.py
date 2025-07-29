import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

CWE_CLASSES = [
    # "normal",  # Non-vulnerable
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


def get_unique_path(base_path):
    """
    取得唯一的檔案路徑，避免覆蓋已存在的檔案。
    :param base_path: 基礎檔案路徑
    :return: 唯一的檔案路徑
    """
    if not os.path.exists(base_path):
        return base_path
    counter = 2
    while True:
        new_path = f"{base_path}_{counter}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def load_data_and_split(json_file, random_state=15, test_size=0.1):
    """
    根據標籤進行分層抽樣，將資料分為訓練集和測試集。
    """
    from sklearn.model_selection import train_test_split
    # 讀取 JSON 資料集
    with open(json_file, 'r') as f:
        dataset = json.load(f)

    labels = [item['function_label'] for item in dataset]  # 提取標籤
    data = [{"code": item["code"], "label": item["function_label"]} for item in dataset]

    # 使用 stratify 參數進行分層抽樣
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, 
        labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    # 再從剩餘的訓練集中切出驗證集
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        train_data, 
        train_labels, 
        test_size=test_size / (1 - test_size), 
        random_state=random_state, 
        stratify=train_labels)
    
    return  train_data, eval_data, test_data
