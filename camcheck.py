#v4l2によるオートフォーカスカメラのマニュアル制御テスト
#q/aでフォーカスざっくり変更、w/sで微動

import cv2
import os

focus=100

#カメラのフォーカス、露出、ホワイトバランスをマニュアル制御化
os.system('v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0')
os.system('v4l2-ctl -d /dev/video0 -c focus_absolute='+str(focus))
os.system('v4l2-ctl -d /dev/video0 -c auto_exposure=1 -c exposure_time_absolute=300')
os.system('v4l2-ctl -d /dev/video0 -c white_balance_automatic=0')
os.system('v4l2-ctl -d /dev/video0 -c white_balance_temperature=4600')

#解像度パラメータ
WIDTH  = 1920  #320/640/800/1024/1280/1920
HEIGHT = 1080   #240/480/600/ 576/ 720/1080

#カメラ入力インスタンス定義
capture = cv2.VideoCapture(0)

#解像度変更
capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if capture.isOpened() is False:
    raise IOError

while(True):

    #画像キャプチャ
    ret, frame = capture.read()
    if ret is False:
        raise IOError
    
    #画像表示
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        focus += 100
        if focus > 1023:
            focus = 1023
        os.system('v4l2-ctl -d /dev/video0 -c focus_absolute='+str(focus))
        print("focus = ", focus)

    elif key == ord("w"):
        focus += 10
        if focus > 1023:
            focus = 1023
        os.system('v4l2-ctl -d /dev/video0 -c focus_absolute='+str(focus))
        print("focus = ", focus)

    elif key == ord("a"):
        focus -= 100
        if focus < 0:
            focus = 0
        os.system('v4l2-ctl -d /dev/video0 -c focus_absolute='+str(focus))
        print("focus = ", focus)

    elif key == ord("s"):
        focus -= 10
        if focus < 0:
            focus = 0
        os.system('v4l2-ctl -d /dev/video0 -c focus_absolute='+str(focus))
        print("focus = ", focus)

    elif key == 27: #ESC
        break
