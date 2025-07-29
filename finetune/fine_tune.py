import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForLanguageModeling, BitsAndBytesConfig, AutoModelForSequenceClassification, CodeLlamaTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from datetime import datetime
import bitsandbytes as bnb
import time
import json
from sklearn.metrics import classification_report
import os
import yaml
import argparse
from utils import CWE_CLASSES, LABEL_TO_CWE, CWE_TO_LABEL, get_unique_path
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(num_labels , model_name="codellama/CodeLlama-7b-Instruct-hf"):
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

# def build_conversation(system_prompt, turns):
#     """
#     <s>
#         [INST]
#             <<SYS>>{{ system_prompt }}<</SYS>>
#         {{ user_msg_1 }}
#         [/INST]
#         {{ model_amswer_1 }}
#     </s>
#     <s>
#         [INST]
#         {{ user_msg_2 }}
#         [/INST]
#         {{ model_amswer_2 }}
#     </s>
#     """
#     conv = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
#     for i, (user_msg, model_reply) in enumerate(turns):
#         conv += f"{user_msg} [/INST] {model_reply}</s>"
#         if i < len(turns) - 1:
#             conv += "<s>[INST] "
#     return conv

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

def tokenize(tokenizer,sample):
    """
    - 將 function 轉換為 Token，並將 function 本身當作 labels 進行自監督學習。
    """

    prompt = build_prompt(sample["code"], sample["buggy_lines"])
    tokenized = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length")

    # input_ids = tokenized["input_ids"].squeeze(0)
    # attention_mask = tokenized["attention_mask"].squeeze(0)
    
    # 將 cwe_id 轉為整數標籤
    # labels = torch.tensor(CWE_TO_LABEL[sample["cwe_id"]], dtype=torch.long)
    tokenized["labels"] = CWE_TO_LABEL[sample["cwe_id"]]
    # labels = input_ids.clone() # Causal LM 要求 labels 與 input_ids 一樣
    # return {
    #     "input_ids": input_ids, 
    #     "attention_mask": attention_mask,
    #     "labels": labels,
    # }
    return tokenized

def setup_qlora(model):
    """
    設定 LoRA 微調配置，應用至模型。

    :param model: 量化後的 Code LLaMA 模型
    :return: 套用 QLoRA 設定後的模型
    """
    # 關閉權重的 requires_grad（不更新原始權重，只更新 LoRA 層）
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8, # Rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS", # 預測 function 的漏洞類型
        # task_type="CAUSAL_LM", # 生成 function 補全或修復漏洞
    )

    model = get_peft_model(model, lora_config)
    
    # 列出目前哪些參數是可訓練的，並計算 可訓練參數的比例
    model.print_trainable_parameters()
    return model

def train_model(model, tokenizer, train_data, test_data, output_dir, config):
    """
    訓練 QLoRA 微調的 Code LLaMA，進行 function 級別漏洞分類。

    :param model: 量化並加上 QLoRA 的模型
    :param tokenizer: Tokenizer
    :param train_data: 訓練集
    :param test_data: 測試集
    :param output_dir: 訓練結果儲存路徑
    """
    model.to(device)

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"], # 單個 GPU 訓練的 batch size
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"], # 梯度累積 32 次，相當於 batch size = 2 * 32 = 64
        warmup_steps=config["warmup_steps"], # 訓練開始前的學習率 Warmup 步數，防止學習率過大導致發散
        eval_strategy=config["eval_strategy"], # 每 N 個 step 進行一次驗證
        save_strategy=config["save_strategy"], # 每 N 個 step 儲存一次 checkpoint
        eval_steps=config["eval_steps"], # 每 50 個 step 進行一次驗證
        save_steps=config["save_steps"], # 每 100 個 step 儲存模型
        logging_steps=config["logging_steps"], # 每 10 個 step 顯示一次 log
        learning_rate=float(config["learning_rate"]), # 設定學習率（比較適合 LoRA 訓練）
        num_train_epochs=config["num_train_epochs"], # 訓練 3 個 epoch
        weight_decay=config["weight_decay"], # L2 正則化，防止過擬合
        bf16=config["bf16"], # 使用 `bfloat16`（適合 RTX 40 系列 GPU）
        optim=config["optim"], # AdamW 8-bit 量化，降低記憶體需求
        logging_dir=config["logging_dir"], # 訓練過程 Log 儲存路徑
        run_name=f"classification-codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # 設定訓練名稱，包含時間戳記
    )

    trainer = Trainer(
        model=model, # 你的 LoRA 訓練模型
        args=train_args, # 訓練參數（TrainingArguments）
        train_dataset=train_data, # 訓練集
        eval_dataset=test_data, # 測試集
        tokenizer=tokenizer, # Tokenizer
        # data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False)
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    )

    # 開始訓練
    model.config.use_cache = False  # QLoRA 訓練時要關閉 cache
    trainer.train()  # 開始訓練
    model.config.use_cache = True  # 訓練完成後重新開啟
    # trainer.train(resume_from_checkpoint="/home/tu/exp/Training/Code_llama/AZOO/result/finetune/QLoRA_Codellama_classification_7b_3/checkpoint-500") # resume_from_checkpoint=True 會從上次儲存的 checkpoint 開始訓練
    

def main():

    config = load_config()

    model_name = config["model_name"]
    
    if config["task"] == "multi":
        num_labels = len(CWE_CLASSES)
    elif config["task"] == "binary":
        num_labels = 2
    else:
        raise ValueError(f"Unsupported task type: {config['task']}. Use 'multi' or 'binary'.")


    print("🚀 載入模型與 tokenizer ...")
    model, tokenizer = load_model(num_labels, model_name)

    print("📂 載入數據集 ...")
    train_data, eval_data, test_data = load_data(
        train_file = config["train_file"],
        eval_file= config["eval_file"],
        test_file= config["test_file"]
    )
    print("⚙️ 設定 tokenizer ...")
    tokenizer = setup_tokenizer(model, tokenizer)

    print("🔄 Tokenizing 訓練/測試數據 ...")
    tokenized_train_data = train_data.map(lambda sample: tokenize(tokenizer, sample), remove_columns=train_data.column_names)
    tokenized_eval_data = eval_data.map(lambda sample: tokenize(tokenizer, sample), remove_columns=eval_data.column_names)
    tokenized_test_data = test_data.map(lambda sample: tokenize(tokenizer, sample), remove_columns=test_data.column_names)
    # tokenized_train_data = train_data.map(
    #     lambda x: {
    #         k: torch.tensor(v, dtype=torch.long) if isinstance(v, list) else v
    #         for k, v in tokenize(tokenizer, x).items()
    #     },
    #     remove_columns=train_data.column_names
    # )

    # tokenized_test_data = test_data.map(
    #     lambda x: {
    #         k: torch.tensor(v, dtype=torch.long) if isinstance(v, list) else v
    #         for k, v in tokenize(tokenizer, x).items()
    #     },
    #     remove_columns=test_data.column_names
    # )

    # tokenized_eval_data = eval_data.map(
    #     lambda x: {
    #         k: torch.tensor(v, dtype=torch.long) if isinstance(v, list) else v
    #         for k, v in tokenize(tokenizer, x).items()
    #     },
    #     remove_columns=eval_data.column_names
    # )
   
    print("🎛️ 設定 QLoRA 模型 ...")
    model = setup_qlora(model)

    # output_file = get_unique_path(config["output_dir"])
    output_file = "/home/tu/exp/Training/Code_llama/AZOO/result/finetune/QLoRA_Codellama_classification_7b_4"
    

    print("🎯 開始訓練 ...")
    start_time = time.time()
    train_model(model, tokenizer, tokenized_train_data, tokenized_eval_data, output_file, config)
    end_time = time.time()
    total_time = end_time - start_time    

    # 儲存訓練時間
    training_time_path = os.path.join(output_file, "config.txt")
    with open(training_time_path, "w") as f:
        f.write(f"Config: {config}\n")
        f.write(f"Training time: {round(total_time, 2)} seconds\n")
    print(f"⏱️ Training time saved to: {training_time_path}")

if __name__ == "__main__":
    main()
    