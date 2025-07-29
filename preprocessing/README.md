# 資料前處理
- 本流程負責將 APK 透過 MobSF 與 Joern 進行靜態分析、擷取函式與漏洞資訊、圖轉換與語意嵌入，產出給 GNN 與 LLM 使用的資料。

## 一、蒐集將被 MobSF 分析出有漏洞的 java 檔
| 檔案 | 說明 |
|------|------|
| `getMobSFResult.py` | ✅ 合併版腳本：執行以下三項流程：<br>1️⃣ 上傳並分析 APK（MobSF API）<br>2️⃣ 擷取並簡化分析報告（如漏洞行、CWE）<br>3️⃣ 使用 `docker cp` 抽取漏洞相關 Java 檔案，儲存至 `tmp_java/`<br>📌 此步會同時產生分析結果與原始碼 |

📌 輸出：
- `result/*.json`：MobSF 的原始分析報告  
- `analysis_output.json`：格式化後的精簡漏洞記錄  
- `tmp_java/{apk}/{file.java}`：與漏洞相關的 Java 原始檔案

## 二、將 java 檔透過 joern 解析 (得到bin檔)
| 檔案 | 說明 |
|------|------|
| `joern_process.py` | 使用 `joern-parse` 將每個 Java 檔轉成 `.bin` 檔 |
| `getCodeByFunction.py` | 開啟 `.bin`，以 `cpg.method.map` 查詢函式資訊<br>擷取原始 Java 中的函式程式碼，根據漏洞行標註為 `label=1` 或 `0`，產出 `result.jsonl` |

## 三、利用 GraphToGraph.sc 提取出漏洞之經 joern 解析過的 bin 檔，提取出 CPG 圖

| 檔案 | 說明 |
|------|------|
| `getGraphInfo.py` | 使用 `funcToGraph.sc` 腳本，在 Joern 中開啟 `.bin`，匯出對應 Graph 結構（JSON） |

📌 輸出：
- `graph/{apk}_{file}.json`：包含節點與邊的 CPG 結構（node type, edge type 等）

## 四、對圖之 node 與 edge 做 embedding
| 檔案 | 說明 |
|------|------|
| `embedding.py` | 使用微調好的 CodeLLaMA 模型，為每個節點產生語意向量（node_type + code）<br>轉換成 GNN 可使用的 `.pt` 結構 |

📌 輸出：
- `graph_with_embedding/*.pt`：PyG 資料，包括：
  - `x`: 節點語意向量  
  - `y`: label（0/1）  
  - `edge_index`, `edge_attr`: 圖結構  
  - `codes`, `lines`, `types`: 原始碼與補充資訊

## 五、vul樣本根據漏洞行切圖
| 檔案 | 說明 |
|------|------|
| `slice_graph.py` | 根據「漏洞行」或「節點類型(CALL/IDIENTIFIER)」進行圖裁切，產出 `sliced_graph` 欄位 |
📌 輸入：
- `*.pt`：包含 `full_graph` 的 pt 檔 (例如: vul_train.pt)
📌 輸出：
- `*.pt`：每筆資料新增 `sliced_graph` 欄位，供後續圖神經網路使用

# 版本需求:
- Joern (4.0.370)