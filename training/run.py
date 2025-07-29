import os
import yaml
import subprocess
from itertools import product
from copy import deepcopy

# 所有組合
graph_types = ["full", "sliced"]
fusion_types = ["llm", "gnn", "concat"]
llm_types = ["ft"]

# base config 載入
with open("/home/tu/exp/EXP_final/training/multi/config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

# 設定儲存位置
config_dir = "/home/tu/exp/EXP_final/training/multi/generated_configs"
os.makedirs(config_dir, exist_ok=True)


# 產生組合
all_configs = []

# llm-only
for llm_type in llm_types:
    all_configs.append({
        "fusion_type": "llm",
        "llm_type": llm_type,
        "gnn_model": "gcn",
        "graph_type": "sliced"
    })


# gnn-only
for graph_type in graph_types:
    all_configs.append({
        "fusion_type": "gnn",
        "llm_type": "ft",
        "graph_type": graph_type
    })


# concat (llm + gnn)
for llm_type, graph_type in product(llm_types, graph_types):
    all_configs.append({
        "fusion_type": "concat",
        "llm_type": llm_type,
        "gnn_model": "gcn",
        "graph_type": graph_type
    })

for config_entry in all_configs:
    fusion_type = config_entry["fusion_type"]
    llm_type = config_entry.get("llm_type", "")
    graph_type = config_entry.get("graph_type", "")

    if fusion_type == "llm":
        run_name = f"{fusion_type}_{llm_type}_llm"
    elif fusion_type == "gnn":
        run_name = f"{fusion_type}_{graph_type}_gcn"
    elif fusion_type == "concat":
        run_name = f"{fusion_type}_{llm_type}_llm_{graph_type}_gcn".strip("_")

    # 使用 auto 模式，一次完成 train + test
    config = deepcopy(base_config)
    config.update(config_entry)
    config["mode"] = "auto"  # 改為 auto 模式

    config_path = os.path.join(config_dir, f"{run_name}_auto.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"🚀 Running: {run_name} [auto]")
    subprocess.run(["python", "/home/tu/exp/EXP_final/training/multi/train.py", "--config", config_path])

    print(f"✅ Done: {run_name}")