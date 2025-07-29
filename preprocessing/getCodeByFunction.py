import os 
import subprocess
import pexpect
import json
import re
from tqdm import tqdm
import logging

# === 設定區 ===
BASE_DIR = "/home/tu/exp/EXP_final/Example_output"
LOG_FILE = os.path.join(BASE_DIR, "output_Function_Code.log")
OUTPUT_JSONL = os.path.join(BASE_DIR, "result.jsonl")
RECORD_FILE = os.path.join(BASE_DIR, "getFunctionCodeRecord.txt")
ANALYSIS_JSON = os.path.join(BASE_DIR, "analysis_output.json")
# SELECTED_FILE_LIST = os.path.join(BASE_DIR, "selected_file.txt")
JAVA_DIR = os.path.join(BASE_DIR, "tmp_java")
BIN_DIR = os.path.join(BASE_DIR, "outputBin")
BLACKLIST_CWE = {"CWE-532", "CWE-919"}

# === Logging 設定 ===
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode="a"),
                              logging.StreamHandler()])


# === 輔助函式 ===
def getMethodInfo(joern, bin_file):
    query = 'cpg.method.map(m => Map("fullName" -> m.fullName,"functionName" -> m.name,"lineNumberStart" -> m.lineNumber,"lineNumberEnd" -> m.lineNumberEnd)).toJsonPretty'
    try:
        joern.sendline(f'importCpg("{bin_file}")')
        joern.expect("joern>")
        joern.sendline(query)
        joern.expect("joern>")
        return joern.before.strip()
    except Exception:
        return None

def clean_ansi_text(text_list):
    """清理包含ANSI轉義序列的文本列表"""
    # 將字符列表組合成字符串
    text = ''.join(text_list)
    
    # 完全移除 ANSI 轉義序列
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-9;]*[a-zA-Z])')
    cleaned_text = ansi_escape.sub('', text)
    
    # 移除殘留的特殊終端機控制符
    non_ansi_chars = re.compile(r'\[\?[0-9;]+[a-zA-Z]')
    cleaned_text = non_ansi_chars.sub('', cleaned_text)

    # 清除隱藏控制符號
    cleaned_text = re.sub(r'[\x1b\x07\x08]', '', cleaned_text)  # 移除 `\x1b`
    
    # 移除多餘的換行與空白字元
    cleaned_text = re.sub(r'\r\n', '\n', cleaned_text)  # 替換 `\r\n` 為 `\n`
    cleaned_text = re.sub(r'\s*\n\s*', '\n', cleaned_text)  # 移除前後空白的換行
    return cleaned_text    


def extract_valid_methods(raw_output):
    #  先找到所有 {...} 小區塊
    object_blocks = re.findall(r"{.*?}", raw_output, re.DOTALL)
    
    valid_methods = []
    
    for block in object_blocks:
        # 檢查有沒有 lineNumberStart 和 lineNumberEnd，且後面是數字
        has_line_start = re.search(r'"lineNumberStart"\s*:\s*\d+', block)
        has_line_end = re.search(r'"lineNumberEnd"\s*:\s*\d+', block)
        
        if has_line_start and has_line_end:
            try:
                method_info = json.loads(block.replace("\n", "").replace("\r", ""))
                valid_methods.append(method_info)
            except Exception as e:
                print(f"Failed to parse block: {block[:30]}..., error: {e}")
    
    return valid_methods


def extract_method_code(file_path, start, end):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        method_code = "".join(lines[start-1:end])
        return method_code
    except Exception as e:
        print(f"Error extracting method code from {file_path} at lines {start}-{end}: {e}")
        return ""

def extract_buggy_code(file_path, buggy_line, cwe_id):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        buggy_code = lines[buggy_line-1]
        return buggy_code
    except Exception as e:
        print(f"Error extracting buggy code from {file_path} at line {buggy_line}: {e}")
        return ""

def load_processed(file):
    if os.path.exists(file):
        with open(file, "r") as f:
            return set(line.strip() for line in f)
    return set()

def record_processed(file, record):
    with open(file, "a") as f:
        f.write(record + "\n")

# === 主邏輯 ===

def main():
    with open(ANALYSIS_JSON, "r") as f:
        analysis_data = json.load(f)
    processed = load_processed(RECORD_FILE)
    # selected_files_set = load_processed(SELECTED_FILE_LIST)
    joern = pexpect.spawn("joern", encoding="utf-8", timeout=6000)
    joern.expect("joern>")

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as fout:
        for item in tqdm(analysis_data, desc="Processing APK files", unit="file"):
            
            apk_name = item["apk_name"]
            file_path = item["file"]
            key = os.path.join(apk_name, file_path)
            cwe_id = item["cwe_id"]
            description = item["description"]
            vul_lines = item["vul_lines"]
            unique_key = f"{apk_name}:{file_path}:{cwe_id}"
            if unique_key in processed or cwe_id in BLACKLIST_CWE:
                continue
            bin_file = os.path.join(BIN_DIR, key.replace("/", "_") + ".bin")
            try:
                logging.info("[+] Processing file: %s", unique_key)
                raw_output = getMethodInfo(joern, bin_file)
                methods = extract_valid_methods(clean_ansi_text(raw_output))
            except Exception:
                joern.sendline("exit")
                joern.close()
                joern = pexpect.spawn("joern", encoding="utf-8", timeout=6000)
                joern.expect("joern>")
                continue

            for method in methods:
                start, end = method.get("lineNumberStart"), method.get("lineNumberEnd")
                if start is None or end is None:
                    continue

                method_code = extract_method_code(os.path.join(JAVA_DIR, apk_name, file_path), start, end)
                buggy_lines = {
                    line: extract_buggy_code(os.path.join(JAVA_DIR, apk_name, file_path), line, cwe_id)
                    for line in vul_lines if start <= line <= end
                }

                result = {
                    "apk_name": apk_name,
                    "file": file_path,
                    "functionName": method.get("functionName"),
                    "fullName": method.get("fullName"),
                    "lineNumberStart": start,
                    "lineNumberEnd": end,
                    "code": method_code,
                    "buggy_lines": buggy_lines,
                    "cwe_id": cwe_id if buggy_lines else "",
                    "description": description if buggy_lines else "",
                    "graph": "",
                    "label": 1 if buggy_lines else 0
                }

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                record_processed(RECORD_FILE, unique_key)

    joern.sendline("exit")
    joern.close()

if __name__ == "__main__":
    main()