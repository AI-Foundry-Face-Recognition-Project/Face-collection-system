## 什麼是 Face-collection-system?
Face-collection-system分為兩個部分
### server.py:伺服器端(負責偵測人臉等功能)
server.py主要透過RetinaFace跟ArcFace來獲取臉部資訊，將監視器中的經過的人進行臉部擷取，並上傳至雲端資料庫。
### main.py:樹梅派端(使用者輸入資料)
main.py主要是設計用於嵌入式裝置樹梅派上運行，該樹梅派須有一可顯示的螢幕來提供使用者告知身分，main.py會將資料庫中尚未辨識的人臉呈現在畫面上，使用者只需輸入學號資訊，就可以將其上傳，做為日後研究的素材。

## 如何安裝 Face-collection-system
建議使用 Python 3.6 創建一個新的虛擬環境並安裝所需的依賴項。

### Face-collection-system
```
git clone https://github.com/AI-Foundry-Face-Recognition-Project/Face-collection-system.git
```

### 安裝所需的套件

## 如何運行 Face-collection-system
1.更正sqlinfo.py，將資料庫資訊填入
2.執行server.py，開始擷取臉部信息
3.在樹梅派上執行main.py