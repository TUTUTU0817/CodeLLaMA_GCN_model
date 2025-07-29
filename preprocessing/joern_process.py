import os 
import subprocess
import json
import logging
from tqdm import tqdm
import concurrent.futures
from collections import defaultdict

# === 設定區 ===
BASE_DIR = "/home/tu/exp/EXP_final/Example_output"
JOERN_PARSE_PATH = "joern-parse"
JAVA_DIR = os.path.join(BASE_DIR, "tmp_java")
BIN_DIR = os.path.join(BASE_DIR, "outputBin")
RECORD_FILE = os.path.join(BASE_DIR, "parsed_files.txt")
MOBSF_ANALYSIS_JSON = os.path.join(BASE_DIR, "analysis_output.json")

BLACKLIST_CWE = {"CWE-532", "CWE-919"}

MAX_PER_APK = 10
MAX_WORKERS = 8
MAX_PER_CWE = 1000


# === Logging ===
logging.basicConfig(level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "output_joern_process.log"), mode="a"),
        logging.StreamHandler()
    ]
)


# === 工具函式 ===

def record_success(file_path, record):
    with open(file_path, "a") as f:
        f.write(record + "\n")

def load_parsed(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return set(line.strip() for line in f)
    return set()

# === 篩選要處理的 Java 檔案 ===

def controll_num(max_per_apk=MAX_PER_APK):
    cwe_count = defaultdict(int)
    running_count = defaultdict(int)
    apk_file_count = defaultdict(int)

    parsed_files = load_parsed(RECORD_FILE)

    with open(MOBSF_ANALYSIS_JSON, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

    for item in analysis_data:
        apk_name = item.get("apk_name")
        file_path = item.get("file")
        cwe = item.get("cwe_id")
        java_path = os.path.join(JAVA_DIR, apk_name, file_path)
        if java_path in parsed_files:
            cwe_count[cwe] += 1

    for cwe_id, count in sorted(cwe_count.items()):
        print(f"{cwe_id}: {count} files processed")

    target_files = set()
    for item in analysis_data:
        apk_name = item.get("apk_name")
        file_path = item.get("file")
        cwe = item.get("cwe_id")
        java_path = os.path.join(JAVA_DIR, apk_name, file_path)

        if (
            cwe in BLACKLIST_CWE or
            cwe_count[cwe] + running_count[cwe] >= MAX_PER_CWE or
            java_path in parsed_files or
            apk_file_count[apk_name] >= max_per_apk
        ):
            continue

        target_files.add(java_path)
        running_count[cwe] += 1
        apk_file_count[apk_name] += 1

    logging.info(f"✅ 本次需要解析 {len(target_files)} 個 Java 檔案")
    return list(target_files)


# === Joern 解析單檔 ===

def joern_parse(joern_parse_path, java_file_path):
    prefix = f"{JAVA_DIR}/"
    output_file_name = java_file_path.replace(prefix, "").replace("/", "_")
    output_path = os.path.join(BIN_DIR, f"{output_file_name}.bin")

    logging.warning(f"正在解析 {java_file_path} → {output_path} ...")
    cmd = [joern_parse_path, java_file_path, "--output", output_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        logging.info(f"✅ 解析完成：{output_path}")
        record_success(RECORD_FILE, java_file_path)
    else:
        logging.error(f"❌ 解析失敗：{java_file_path}")
        logging.error(result.stderr)

# === Joern 多檔平行解析 ===

def joern_parse_all(joern_parse_path, target_files, max_workers=MAX_WORKERS):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(joern_parse, joern_parse_path, file_path)
            for file_path in target_files
        ]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parsing Java", unit="file"):
            _ = f.result()


# === 主程式 ===

def main():
    target_files = controll_num()
    joern_parse_all(JOERN_PARSE_PATH, target_files)

if __name__ == "__main__":
    main()