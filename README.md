# AIcup
 
若要進行訓練，直接執行AIcup.ipynb即可
# 建議配置環境
作業系統為Ubuntu 22.04.3 LTS
| module | version |
| ------ | ------- |
| numpy | 1.25.2 |
| tqdm | 4.65.0 |
| torch |2.1.1+cu118|
|torchvision|0.16.1+cu118 |
|torchsummary|1.5.1|
|cv2| 4.8.1.78 |
|pysodmetrics| 1.4.2 |

若顯存不夠(約需20GB)，可能會在運行時報錯OUT_OF_MEMORY，解決方式為將第六個儲存格定義dataset時參數to_gpu設為False
# 檔案說明
AIcup.ipynb：主要訓練，若想用已有權重inference可至submitting區塊直接執行  
test.ipynb：測試用程式碼、summary模型架構  
model.py：定義模型  
util.py：自訂transforms、loss等  
