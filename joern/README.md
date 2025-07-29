# Joern 指令介紹(我有使用到的) 
## joern-parse 解析程式碼
- `./joern/joern-cli/joern-parse -- language java [存放java檔之目錄]`
    - 會生成 `cpg.bin`
## joern-export 導出圖
- `./joern/joern-cli/joern-export cpg.bin --repr cpg14 --out output/`
    - 生成 .dot 檔（以 function 為單位）
## 轉成 PNG 圖
- `dot -Tpng ./output/0-cpg.dot -o output_graph.png`
## 找出所有method之行數
- `cpg.file.name("UDecoder.java").method.lineNumber.l`
    - val res12: List[Int] = List(35, 42, 50, 101, 109, 155, 163, 186, 191, 246, 252, 261, 271)
## 根據 fullName 導出圖 
- cpg.method.fullNameExact("XXX").dotCpg14.l
    - 根據funciton 的 fullname (唯一key) 生成對應的圖


# scala 腳本檔介紹
## graph-for-funcs.sc (其他研究)
- 從現有研究之公開程式碼檔中下載
- 現有研究主要使用此腳本自動生成每個檔案之cpg
- 由於 joern 版本不符，我無法使用(試過降版本，還是無法)，因此模仿寫出自己的「funcToGraph.sc」
## funcToGraph.sc (我自己寫的)
- 根據經過 joern-process.py 產生的 bin 檔，去生成每個java 的的圖存在"/home/tu/exp/EXP_final/Dataset/multi/graph"