# AIcup
此為AI cup 2024春季賽「以生成式AI建構無人機於自然環境偵察時所需之導航資訊競賽 II － 導航資料生成競賽」之相關程式碼  
若要進行訓練，直接執行AIcup.ipynb即可  
若想載入權重進行inference，至submiting區塊輸入權重檔及取消註解，依序執行第1,2,5,8,10個cell  
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
# 文件配置  
```
├AIcup.ipynb
├test.ipynb
├model.py
├util.py
├─Training_dataset  
│  ├─img  
|  |  ├xxx.jpg
|  |  ├ ...
│  └─label_img
|     ├xxx.png  
|     ├ ...
├Testing_dataset
|  ├ xxx.jpg
|  ├ ...
├Private_dataset
|  ├xxx.jpg
|  ├ ...
├weights
|  ├xxx.pt
|  ├xxx_.pt
|  ├ ...
├submit
|  ├submit-xx-xx
|  ├submit-xx-xx.zip
|  ├ ...
├tmp
|  ├xxx.png
|  ├ ...
```
weights, tmp, submit分別為程式為輸出權重、暫存圖片、模型輸出結果之資料夾，開始前確定其存在以免報錯
注意weights中權重會有兩版本  
xxx.pt為`torch.jit.script(model)).save()`之權重  
xxx_.pt為`torch.save(model.state_dict())`之權重  
使用之間差異可見https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-and-loading-models
