# 概要

春のパン祭りシール台紙を撮影して、点数を自動集計します。  
面倒な手計算とおさらば！

[](https://github.com/hubnoki/harupan/blob/master/doc_images/harupan_example_230415b.gif)


**----注意----**

**ほぼ必ず点数認識間違いを含むので、認識結果の目視チェックと点数の補正が必須です。**

<br/>

# 使い方

1. カメラの付いたAndroidまたはiOS端末にDroidCamアプリをインストールする。  
   [DroidCam - Webcam for PC - Google Play のアプリ](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam&hl=ja&gl=US&pli=1)  
   [「DroidCam Webcam & OBS Camera」をApp Storeで](https://apps.apple.com/jp/app/droidcam-webcam-obs-camera/id1510258102)
2. PCと上記端末を同じWiFiルータに接続しておく。
3. 上記端末でDroidCamを起動、PCで本アプリを起動。
4. 本アプリで、DroidCam画面に表示されるIPアドレスとポート番号を入力。
5. "Connect"ボタンをクリックするとDroidCamに接続され、画像取得と点数自動計算が行われる。
6. "Stop"ボタンをクリックすると、画像の更新が一旦停止する。"Resume"ボタンをクリックすると、更新が再開する。

その他。

- ウィンドウサイズを変更すると、表示画像サイズも追従します。見やすいサイズに調整してください。

<br/>

# exe生成方法(Pyinstaller, Windows)

```
pyinstaller harupan.py --add-data data\harupan_svm_220412.dat;data --add-data data\templates2021.json;data --noconsole --onefile

```



