# multi (9類訓練)
# 資料集
- /home/tu/exp/EXP_final/Dataset/multi/vul_train.pt
- /home/tu/exp/EXP_final/Dataset/multi/vul_eval.pt
- /home/tu/exp/EXP_final/Dataset/multi/vul_test.pt

# config 設定
- task: 
    - 二分類任務 / 多分類任務  
    - 參數
        - multi
        - binary
- llm_type: 
    - 原始 codellama / 微調codellama
    - 參數
        - ft
        - nft
- graph_type:
    - 原始圖 / 裁切圖
    - 參數
        - sliced
        - full
- gnn_model:
    - gnn 模型類型
    - 參數
        - gcn
        - ggnn
        - gat
- gnn_out_dim:
    - gnn 輸出維度
    - 參數
        - 64
        - 128
        - 256...
- fusion_type:
    - 融合 / gnn-only / llm-only
    - 參數
        - concat
        - gnn
        - llm
- mode:
    - 訓練模式/測試模式/自動訓練+測試
    -  參數
        - train
        - test
        - auto
- num_trails:
    - 實驗重複次數	
        - 參數
            - int
- batch_size:
    - 批次大小
    - 參數
        - 8
- epochs: 
    - 訓練輪數	
    - 參數
        - 50
- lr: 
    - 學習率	
    - 參數
        - 5e-5
- warmup_steps:
    - 預熱步數
    - 參數
        - 100
- llm_path: 
    - 微調 codellama 路徑
    - 參數
        - "/home/tu/exp/Training/Code_llama/AZOO/result/model/QLoRA_Codellama_classification_7b_3/checkpoint-1797"
- output_dir: 
    - 模型輸出結果以及各項參數紀錄路徑
    - 參數
        - "/home/tu/exp/EXP_final/training/multi/result/nine_classes" 

# utils.py
- 重複使用函數
# run
- 自動執行 各種融合實驗組合
# 執行命令
- python /home/tu/exp/EXP_final/training/multi/train.py --config /home/tu/exp/EXP_final/training/multi/config.yaml