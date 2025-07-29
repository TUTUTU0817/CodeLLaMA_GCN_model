# Dataset 介紹
- 主要分為 binary(漏洞/非漏洞) 跟 multi(9類漏洞)，我在論文中只有報告 multi(9類漏洞)。

## multi(9類漏洞)
### graph 資料夾
    - 包含了實驗需要的每張圖，以 json 格式儲存(以 java 檔案為單位)
    - 資料格式
        - nodes
            - id: 節點唯一標識符
            - code: 程式碼片段
            - type: 節點類型
            - fullName: 完整名稱(每個function的唯一值，可以利用這個來查看此function的圖)
            - lineNumber: 行號
        - edges
            - source: 來源節點 ID
            - target: 目標節點 ID
            - type: 邊的類型
### vul_train.pt / vul_eval.pt / vul_test.pt
- 主要訓練模型之資料
- 資料格式
    - 'key': 自建之唯一值，為 apk_name + file + fullName + lineNumberStart
    - 'apk_name': 來自哪個 apk 檔
    - 'file': 檔案在apk中的路徑
    - 'cwe_id': 漏洞CWE_ID
    - 'description': 漏洞描述
    - 'functionName': 函式名稱
    - 'fullName': 完整名稱
    - 'lineNumberStart': 程式碼在檔案中的起始行
    - 'lineNumberEnd': 程式碼在檔案中的結束行
    - 'buggy_lines': {漏洞行號: 該行程式碼}
    - 'code': 此 function 程式碼
    - 'label': 此 function 是否有漏洞 
    - 'full_graph': 此 function 原始圖 (pyg 格式，已經經過初始化 embedding)
    - 'sliced_graph_buggy': 此 function 經漏洞行裁切圖 (pyg 格式，已經經過初始化 embedding)
    - 'cls_embed': 此 function 經過「原始 code llama」當作 encoder 的 embedding 
    - 'ft_cls_embed': 此 function 經過「微調 code llama」當作 encoder 的 embedding

### train_tmp_data / eval_tmp_data / test_tmp_data
- 與上述同
- 差異
    - 'sliced_graph_node_type': 此 function 經 CALL/IDIENTIFIER 節點裁切圖 (pyg 格式，已經經過初始化 embedding) 

