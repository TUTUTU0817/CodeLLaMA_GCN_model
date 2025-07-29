# run_pipeline.py
import os
import json
import time
import subprocess
import logging
import requests
import re
import pexpect
from tqdm import tqdm
from requests_toolbelt.multipart.encoder import MultipartEncoder

# === 設定區 ===
APK_DIR = "/home/tu/exp/EXP_final/Example_APK"
OUTPUT_DIR = "/home/tu/exp/EXP_final/Example_output"
MOBSF_URL = "http://localhost:8000"
API_KEY = "2b12a9b9e110551d3380a0fe5560f23d97cab1f829886cb9cbb720ec14bc7eb9"
DOCKER_CONTAINER = "212e0b559097"
UPLOADS_PATH = "/home/mobsf/.MobSF/uploads"

# === 輸出資料路徑 ===
REPORT_DIR = os.path.join(OUTPUT_DIR, "result")
JAVA_DIR = os.path.join(OUTPUT_DIR, "tmp_java")
ANALYSIS_OUTPUT = os.path.join(OUTPUT_DIR, "analysis_output.json")
RESULT_JSONL = os.path.join(OUTPUT_DIR, "result.jsonl")
SELECTED_FILE_TXT = os.path.join(OUTPUT_DIR, "selected_file.txt")
RECORD_TXT = os.path.join(OUTPUT_DIR, "getFunctionCodeRecord.txt")
BIN_DIR = os.path.join(OUTPUT_DIR, "outputBin")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(JAVA_DIR, exist_ok=True)
os.makedirs(BIN_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(OUTPUT_DIR, "pipeline.log"), mode="a"),
                        logging.StreamHandler()
                    ])

def upload_to_mobsf(apk_path):
    multipart_data = MultipartEncoder(fields={'file': (os.path.basename(apk_path), open(apk_path, 'rb'), 'application/octet-stream')})
    headers = {'Content-Type': multipart_data.content_type, "Authorization": API_KEY}
    resp = requests.post(f"{MOBSF_URL}/api/v1/upload", data=multipart_data, headers=headers)
    return resp.json()["hash"]

def scan_apk(scan_hash):
    headers = {"Authorization": API_KEY}
    requests.post(f"{MOBSF_URL}/api/v1/scan", data={"hash": scan_hash}, headers=headers)

def get_json_report(scan_hash, dest_path):
    headers = {"Authorization": API_KEY}
    resp = requests.post(f"{MOBSF_URL}/api/v1/report_json", data={"hash": scan_hash}, headers=headers)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

def docker_cp_java(scan_hash, apk_file):
    with open(ANALYSIS_OUTPUT, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

    # 找出這個 APK 名稱對應的檔案們
    file_paths = [item["file"] for item in analysis_data if item["apk_name"] == apk_file]

    for file_path in file_paths:
        container_path = os.path.join(UPLOADS_PATH, scan_hash, "java_source", file_path)
        local_path = os.path.join(JAVA_DIR, apk_file, file_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        subprocess.run(["sudo", "docker", "cp", f"{DOCKER_CONTAINER}:{container_path}", local_path], check=False)

def scan_all_apks():
    for apk_file in os.listdir(APK_DIR):
        if not apk_file.endswith(".apk"):
            continue
        json_path = os.path.join(REPORT_DIR, f"{apk_file}.json")
        if os.path.exists(json_path):
            logging.info(f"[SKIP] {apk_file} already scanned.")
            continue
        try:
            apk_path = os.path.join(APK_DIR, apk_file)
            logging.info(f"[+] Scanning {apk_file} ...")
            scan_hash = upload_to_mobsf(apk_path)
            scan_apk(scan_hash)
            get_json_report(scan_hash, json_path)
            docker_cp_java(scan_hash, apk_file)
        except Exception as e:
            logging.error(f"[ERROR] Failed {apk_file}: {e}")

def parse_findings():
    all_results = []
    for json_file in os.listdir(REPORT_DIR):
        if not json_file.endswith(".apk.json") and not json_file.endswith(".json"):
            continue
        path = os.path.join(REPORT_DIR, json_file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        apk_name = data.get("file_name", json_file.replace(".json", ""))
        findings = data.get("code_analysis", {}).get("findings", {})
        for _, info in findings.items():
            cwe_id = info.get("metadata", {}).get("cwe", "").split(":")[0].strip()
            desc = info.get("metadata", {}).get("description", "")
            for file, line_str in info.get("files", {}).items():
                lines = [int(l.strip()) for l in line_str.split(",") if l.strip().isdigit()]
                all_results.append({
                    "apk_name": apk_name,
                    "file": file,
                    "cwe_id": cwe_id,
                    "vul_lines": lines,
                    "description": desc
                })
    with open(ANALYSIS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

def clean_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-9;]*[a-zA-Z])')
    return ansi_escape.sub('', text)

def extract_valid_methods(raw_output):
    blocks = re.findall(r"{.*?}", raw_output, re.DOTALL)
    result = []
    for block in blocks:
        if 'lineNumberStart' in block and 'lineNumberEnd' in block:
            try:
                result.append(json.loads(block.replace("\n", "")))
            except: pass
    return result

def extract_code_lines(java_file, start, end):
    with open(java_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[start-1:end])

def extract_buggy_code(java_file, line):
    try:
        with open(java_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines[line-1].strip()
    except:
        return ""

def generate_function_dataset():
    with open(ANALYSIS_OUTPUT, "r") as f:
        data = json.load(f)
    joern = pexpect.spawn("joern", encoding="utf-8", timeout=6000)
    joern.expect("joern>")
    fout = open(RESULT_JSONL, "a", encoding="utf-8")
    processed = set()
    if os.path.exists(RECORD_TXT):
        with open(RECORD_TXT) as f:
            processed = set(line.strip() for line in f)
    for item in tqdm(data, desc="Joern"):
        key = f"{item['apk_name']}:{item['file']}:{item['cwe_id']}"
        if key in processed:
            continue
        bin_file = key.replace("/", "_") + ".bin"
        joern.sendline(f'open("{bin_file}")')
        joern.expect("joern>")
        joern.sendline('cpg.method.map(m => Map("fullName" -> m.fullName,"functionName" -> m.name,"lineNumberStart" -> m.lineNumber,"lineNumberEnd" -> m.lineNumberEnd)).toJsonPretty')
        joern.expect("joern>")
        methods = extract_valid_methods(clean_ansi(joern.before))
        for m in methods:
            s, e = m["lineNumberStart"], m["lineNumberEnd"]
            code_path = os.path.join(JAVA_DIR, item["apk_name"], item["file"])
            code = extract_code_lines(code_path, s, e)
            buggy = {str(l): extract_buggy_code(code_path, l) for l in item["vul_lines"] if s <= l <= e}
            is_vul = len(buggy) > 0
            fout.write(json.dumps({
                "apk_name": item["apk_name"],
                "file": item["file"],
                "functionName": m["functionName"],
                "fullName": m["fullName"],
                "lineNumberStart": s,
                "lineNumberEnd": e,
                "code": code,
                "buggy_lines": buggy,
                "cwe_id": item["cwe_id"] if is_vul else "",
                "description": item["description"] if is_vul else "",
                "label": 1 if is_vul else 0,
                "graph": ""
            }, ensure_ascii=False) + "\n")
        with open(RECORD_TXT, "a") as rf:
            rf.write(key + "\n")
    fout.close()
    joern.sendline("exit")
    joern.close()

def main():
    scan_all_apks()
    parse_findings()
    # generate_function_dataset()

if __name__ == "__main__":
    main()
