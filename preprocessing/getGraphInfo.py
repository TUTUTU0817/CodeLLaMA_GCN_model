import json
import threading
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import time
import torch
from collections import defaultdict
from tqdm import tqdm
import logging
import os
import subprocess
import pandas as pd


# === 路徑設定 ===
BASE_DIR = "/home/tu/exp/EXP_final/Example_output"
JSON_PATH = os.path.join(BASE_DIR, "result.jsonl")
INPUT_DIR = os.path.join(BASE_DIR, "outputBin")
OUTPUT_DIR = os.path.join(BASE_DIR, "graph")
LOG_PATH = os.path.join(BASE_DIR, "output_get_graph_info.log")
JOERN_PATH = "joern"
SCRIPT_PATH = "/home/tu/exp/EXP_final/joern/funcToGraph.sc"


# === Logging 設定 ===
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_PATH, mode="a"),
                              logging.StreamHandler()])

# === 輔助函式 ===
def filter_processed_files(bin_files, output_dir):
    unprocessed, already_processed = [], []
    for bin_file in bin_files:
        output_file = bin_file.replace('.java.bin', '.json') if bin_file.endswith('.java.bin') else bin_file.replace('.bin', '.json')
        output_path = os.path.join(output_dir, output_file)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            already_processed.append(bin_file)
        else:
            unprocessed.append(bin_file)
    return unprocessed, already_processed


def joern_create_graph_single(joern_path, input_bin_path, output_dir, bin_file, script_path):
    """
    自動化處理已經 joern-parse 過的 bin 文件
    
    Args:
        joern_path: joern 可執行文件路徑
        input_bin_path: 包含 .bin 文件的目錄
        output_dir: 輸出 .pt 文件的目錄  
        bin_files: 要處理的 bin 文件列表
        script_path: funcToGraph.sc 的路徑
    """
    
    # 啟動 joern 進程
    joern_process = subprocess.Popen(
        [joern_path], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
    processed_files = []
    
    try:
        # for bin_file in tqdm(bin_files, desc="Processing bin files"):
            
        # 1. 構建完整路徑
        bin_full_path = os.path.abspath(os.path.join(input_bin_path, bin_file))
        
        # 2. 檢查文件是否存在
        if not os.path.exists(bin_full_path):
            logging.warning(f"Warning: {bin_full_path} not found, skipping...")
            return None
        
        # 3. 生成輸出文件名 (去掉 .java.bin 後綴，加上 .pt)
        if bin_file.endswith('.java.bin'):
            output_name = bin_file.replace('.java.bin', '.json')
        elif bin_file.endswith('.bin'):
            output_name = bin_file.replace('.bin', '.json')
        else:
            output_name = bin_file + '.json'
        
        output_full_path = os.path.abspath(os.path.join(output_dir, output_name))

        logging.info(f"Input: {bin_full_path}")
        logging.info(f"Output: {output_full_path}")
        
        # 4. 執行 joern 命令序列
        try:
            logging.info(f'\n=== BEGIN processing {bin_file} ===')
            
            # Step 1: importCpg
            import_cmd = f"importCpg(\"{bin_full_path}\")\n"
            logging.info(f"Executing: {import_cmd.strip()}")
            joern_process.stdin.write(import_cmd)
            joern_process.stdin.flush()
            time.sleep(1)

            
            # Step 2: 設置 outputPath 變數
            output_var_cmd = f'val outputPath = "{output_full_path}"\n'
            logging.info(f"Executing: {output_var_cmd.strip()}")
            joern_process.stdin.write(output_var_cmd)
            joern_process.stdin.flush()
            time.sleep(0.5)
            
            # Step 3: 載入並執行腳本
            script_cmd = f':load {script_path}\n'
            logging.info(f"Executing: {script_cmd.strip()}")
            joern_process.stdin.write(script_cmd)
            joern_process.stdin.flush()
            
            
            # 等待腳本執行完成
            time.sleep(5)
            
            # Step 4: 清理當前 CPG
            delete_cmd = 'delete\n'
            logging.info(f"Executing: {delete_cmd.strip()}")
            joern_process.stdin.write(delete_cmd)
            joern_process.stdin.flush()
            
            # 等待清理完成
            time.sleep(1)
            
            processed_files.append(output_name)
            logging.info(f"✅ Successfully processed {bin_file} -> {output_name}")
        
        except Exception as e:
            logging.error(f"❌ Error processing {bin_file}: {str(e)}")
            return None

    finally:
        logging.info("=== END processing ===")     
        # 關閉 joern 進程
        try:
            joern_process.stdin.write('exit\n')
            joern_process.stdin.flush()
            joern_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            joern_process.kill()
            joern_process.wait()
    
    return processed_files
        
def batch_process_bins(json_path, joern_path, input_dir, output_dir, script_path, skip_processed=True):
    """
    批次處理目錄中的所有 bin 文件
    
    Args:
        joern_path: joern 安裝路徑 (例如: "joern-cli/joern")
        input_dir: 包含 .bin 文件的目錄
        output_dir: 輸出 .pt 文件的目錄
        script_path: funcToGraph_fixed.sc 的完整路徑
    """
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加載 result.jsonl 文件
    filtered_bin_files = []
    try:
        # 讀取 JSON 文件
        # df = pd.read_json(json_path)
        # df['bin_file_name'] = df.apply(
        #     lambda row: f"{row['file_path']}.java.bin", axis=1
        # )
        # 讀取 JSONL 文件
        df = pd.read_json(json_path, lines=True)
        df_filtered = df[df['cwe_id']!=""]  # 過濾 cwe 欄位不是空的記錄
        df_filtered['bin_file_name'] = df_filtered.apply(
            lambda row: f"{row['apk_name']}_{row['file'].replace('/', '_')}.bin", axis=1
        )
        filtered_bin_files = df_filtered['bin_file_name'].tolist()
    
    except Exception as e:
        logging.error(f"❌ Error reading result.jsonl file: {str(e)}")
        return []
    
    
    # 找到所有 bin 文件
    all_bin_files = [f for f in os.listdir(input_dir) if f.endswith('.bin')]
    
    if not all_bin_files:
        logging.warning(f"No .bin files found in {input_dir}")
        return []
    
    logging.info(f"Found {len(all_bin_files)} total bin files")
    
    
    # 過濾 bin 文件，僅保留在 filtered_bin_files 中的文件
    bin_files_to_process = [f for f in all_bin_files if f in filtered_bin_files]
    if not bin_files_to_process:
        logging.info("🎉 No matching bin files to process based on result.jsonl!")
        return []
    
    logging.info(f"Filtered {len(bin_files_to_process)} bin files to process based on result.jsonl")
    
    
    # 過濾已處理的文件
    if skip_processed:
        bin_files, already_processed = filter_processed_files(bin_files_to_process, output_dir)
        if not bin_files:
            logging.info("🎉 All files already processed! Nothing to do.")
            return already_processed
    else:
        bin_files = bin_files_to_process
        already_processed = []
    
    newly_processed = []

    for bin_file in tqdm(bin_files, desc="Processing bin files"):
        logging.info(f"Processing file: {bin_file}")
        try:
            # 處理剩餘的文件
            newly_processed_single = joern_create_graph_single(
                joern_path=joern_path,
                input_bin_path=input_dir,
                output_dir=output_dir,
                bin_file=bin_file,
                script_path=script_path
            )
            
            if newly_processed_single:
                newly_processed.extend(newly_processed_single)
        except Exception as e:
            logging.error(f"❌ Error processing {bin_file}: {str(e)}")
            continue                
            
    all_processed = already_processed + newly_processed

    logging.info(f"\n🎉 Processing complete!")
    logging.info(f"📊 Summary:")
    logging.info(f"  Previously processed: {len(already_processed)}")
    logging.info(f"  Newly processed: {len(newly_processed)}")
    logging.info(f"  Total processed: {len(all_processed)}")
    
    return all_processed


if __name__ == "__main__":
    result = batch_process_bins(
        json_path=JSON_PATH,
        joern_path=JOERN_PATH,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        script_path=SCRIPT_PATH,
        skip_processed=True
    )
    logging.info(f"Processed {len(result)} files.")
