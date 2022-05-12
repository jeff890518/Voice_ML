# Voice_ML
youtube：https://youtu.be/NNAmlbxG__4

基於邊緣運算的概念，在各個道路上設置我們的獨立裝置，直接在救護車經過的道路現場進行第一時間的收音並計算出我們所要的資訊，並透過網路連結伺服器端與邊緣裝置，就能實現隨時隨地優化警笛聲辨識的模型。

裝置使用Raspberry Pi 4為主體，並使用Respeaker-4-mic-array和一個獨立麥克風。利用獨立麥克風接收救護車聲音，將聲音樣本透過都卜勒效應(Doppler　effect)及傅立葉轉換(Fourier transform)建置模型及訓練，再匯入裝置並裝設於道路上。先使用獨立麥克風辨識救護車聲音，再使用Respeaker-4-mic-array進行聲音方向辨識(DOA)，及時偵測警笛聲及救護車方向並透過顯示螢幕提醒用路人。
