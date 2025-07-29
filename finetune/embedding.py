import argparse
import gc
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml
from utils import CWE_CLASSES, CWE_TO_LABEL
import torch.nn as nn
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    return model, tokenizer

def load_fine_tuned_model(num_classes, model_name, checkpoint_path):
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
        num_labels=num_classes,
        quantization_config=quantization_config,
        device_map={"": 0}  
        # device_map="auto"  # 自動分配到可用的 GPU
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    return model, tokenizer

def setup_tokenizer(model, tokenizer):
    """
    設定 tokenizer 以適應 function 級別的訓練資料。
    """
    tokenizer.pad_token = tokenizer.eos_token  # 🔹 把 `pad_token` 設定成 `eos_token`
    tokenizer.pad_token_id = tokenizer.eos_token_id  # 🔹 確保 `pad_token_id` 設定正確
    tokenizer.padding_side = "left"  # Code Llama 預設左側 `padding`
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def build_prompt(code):
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
                    [Response]:
                """
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_msg} [/INST]</s>" 

def tokenize(tokenizer,sample):
    """
    - 將 function 轉換為 Token，並將 function 本身當作 labels 進行自監督學習。
    """

    prompt = build_prompt(sample["code"])
    tokenized = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length", return_tensors="pt")
    # if sample["cwe_id"] == "":
    #     sample["cwe_id"] = "normal"
    
    tokenized["labels"] = CWE_TO_LABEL[sample["cwe_id"]]
    return tokenized

class LLMModel(nn.Module):
    """LLM model for extracting hidden states.
    - 參數: 
        - llm_encoder: LLM 模型
        - config: LLM 配置參數
        - tokenizer: 分詞器
    """
    def __init__(self, llm_encoder, tokenizer):
        super().__init__()
        self.llm_encoder = llm_encoder # LLM 
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, input_embed=None):
        with torch.no_grad():
            if input_ids is not None:
                self.llm_encoder.eval() # 不需要訓練 LLM
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
                outputs = self.llm_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            else:
                outputs = self.llm_encoder(inputs_embeds=input_embed, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            attention_hidden_states = hidden_states[1:]
            final_attention_states = attention_hidden_states[-1]
        return final_attention_states[:, 0, :]


class LLMGNNDataset(Dataset):
    """
    - 參數
        - pt_file: pt 檔案路徑
        - tokenizer: Hugging Face 的 tokenizer
    """
    def __init__(self, pt_file, tokenizer):
        
        entries = torch.load(pt_file)
        self.entries = entries
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        sample = self.entries[idx]
        tokenized = tokenize(self.tokenizer, sample)

        # prompt = build_prompt(sample["code"])
        # tokenized = self.tokenizer(prompt, truncation=True, max_length=1024, padding="max_length", return_tensors="pt")
        # key = f"{sample['apk_name']}_{sample['file'].replace('/', '_').replace('.java','')}_{sample['fullName'].replace('/', '_').replace(':', '_').replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '_')}_{sample['lineNumberStart']}",
        return {
            "key": sample["key"],
            "input_ids": tokenized["input_ids"].squeeze(0),  # 移除 batch 維度
            "attention_mask": tokenized["attention_mask"].squeeze(0),  # 移除 batch 維度
            "label": tokenized["labels"],
        }

def fusion_collate_fn(samples):
    """
    - 參數
        - samples: 一個batch的資料
    """
    keys = [s['key'] for s in samples]  # 收集所有樣本的 key
    input_ids = torch.stack([s['input_ids'] for s in samples])
    attention_mask = torch.stack([s['attention_mask'] for s in samples])
    labels = torch.tensor([s['label'] for s in samples], dtype=torch.long)
    device = input_ids.device
    return {
        "key": keys,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels.to(device),
    }


def extract_embedding(llm_model, ft_llm_model,
                      data_loader, output_file):
    llm_model.eval()
    ft_llm_model.eval()
                
    processed = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):

            key = batch["key"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            llm_hidden_states = llm_model(input_ids=input_ids)
            cls_embed = llm_hidden_states

            ft_llm_hidden_states = ft_llm_model(input_ids=input_ids)
            ft_cls_embed = ft_llm_hidden_states

            for i in range(len(labels)):
                processed.append({
                    "key": key[i],
                    "ft_cls_embed": ft_cls_embed[i].cpu(),
                    "cls_embed": cls_embed[i].cpu(),
                    "label": labels[i].item()
                })

            torch.cuda.empty_cache()
            gc.collect()

    torch.save(processed, output_file)
    print(f"✅ Saved {len(processed)} embeddings to {output_file}")
    

def main():
    config = load_config()
    model_name = "codellama/CodeLlama-7b-Instruct-hf"

    print("🔍 載入 LLM...")
    if config["task"] == "multi":
        num_labels = len(CWE_CLASSES)
        checkpoint_path = config["checkpoint_path"]        
        model, tokenizer = load_model(num_labels, model_name)
        ft_model, ft_tokenizer = load_fine_tuned_model(num_labels, model_name, checkpoint_path)
    elif config["task"] == "binary":
        num_labels = 2
        checkpoint_path = config["llm_path_binary"]
        model, tokenizer = load_model(num_labels, model_name)
        ft_model, ft_tokenizer = load_fine_tuned_model(num_labels, model_name, checkpoint_path)
    else:
        raise ValueError("Unsupported task type. Please use 'multi' or 'binary'.")

    print("⚙️ 設定 tokenizer ...")
    tokenizer = setup_tokenizer(model, tokenizer)
    ft_tokenizer = setup_tokenizer(ft_model, ft_tokenizer)

    llm_model = LLMModel(model, tokenizer).to(device)
    ft_llm_model = LLMModel(ft_model, ft_tokenizer).to(device)

    print("📂 載入訓練/測試資料...")

    # dataset = LLMGNNDataset(config[f"all_data_file_{config['task']}"], tokenizer=tokenizer)
    dataset = LLMGNNDataset(config[f"all_data_file"], tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=fusion_collate_fn)
    extract_embedding(llm_model, ft_llm_model,
                      data_loader, config["emb_output_dir"])

if __name__ == "__main__":
    main()