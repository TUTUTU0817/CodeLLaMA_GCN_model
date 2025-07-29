# 微調 LLM
## 相關技術使用
- QLoRA
## 任務
- multi: 多分類任務
- binary: 二分類任務
## 參數調整
- 可至 config.yaml 檔調整
## 資料集格式
- pt 檔
- 資料集欄位 
    - 'apk_name': 'abdo.free.remote.samsung.tv.apk', 
    - 'file': 'com/loopj/android/http/SimpleMultipartEntity.java', 
    - 'cwe_id': '', 
    - 'description': '', 
    - 'functionName': 'isRepeatable', 
    - 'fullName': 'com.loopj.android.http.SimpleMultipartEntity.isRepeatable:boolean()', 
    - 'lineNumberStart': 132, 
    - 'lineNumberEnd': 135, 
    - 'buggy_lines': [], 
    - 'code': '    @Override // org.apache.http.HttpEntity\n    public boolean isRepeatable() {\n        return false;\n    }\n', 'graph': '', 'label': 0, 
    - 'key': 'abdo.free.remote.samsung.tv.apk_com_loopj_android_http_SimpleMultipartEntity_com.loopj.android.http.SimpleMultipartEntity.isRepeatable_boolean___132'
## 其他
- utils: 將重複功能寫一起