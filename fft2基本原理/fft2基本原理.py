import cv2
import numpy as np
from matplotlib import pyplot as plt

# 0.读取视频文件
cap = cv2.VideoCapture('./video/bike.mp4')

# 循环播放视频文件，同时显示原视频及其对应的灰度图
while cap.isOpened():
    # 逐帧读取视频，ret 为布尔值，表示是否成功读取帧，frame 为当前帧的图像数据
    ret, frame = cap.read()
    if not ret:
        print("没有内容，退出啦 :) ")
        break

    # 1 使用 cv2.cvtColor() 将当前帧的彩色图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2 获得傅里叶系数
    fCoef = np.fft.fft2(gray)

    #print(fCoef)
    #break          #获取一帧测试 ：成功

    # 3 放大傅里叶系数，获得幅度谱(振幅谱）
    mag_spec = 20 * np.log(np.abs(fCoef)+1)   # 另一种常见写法: mag_spec = np.log(np.abs(fCoef))
    
    #print(mag_spec)
    #break              #获取一帧测试 ：成功

    #  通过逆傅里叶变换还原图像
    #img_rc = np.fft.ifft2(fCoef)

    # 4 将振幅谱转换为灰度图像
    mag_spec = cv2.cvtColor(np.uint8(mag_spec), cv2.COLOR_GRAY2BGR)


    # 5 将低频部分移至中心
    fCoef_shifted = np.fft.fftshift(fCoef)  

    # 6 经过fftshift后的振幅谱
    mag_spec_shift = 20 * np.log(np.abs(fCoef_shifted)+1)

    # 7 将fft.fftshift振幅谱转换为灰度图像
    mag_spec_shift = cv2.cvtColor(np.uint8(mag_spec_shift), cv2.COLOR_GRAY2BGR)

    # 将原视频和振幅谱拼接在一起
    #frame = np.hstack((frame, mag_spec))

    # 8 在名为 "gray" 的窗口中显示灰度图像
    cv2.imshow('mg_spc', mag_spec)

    # 9 在名为 "gray" 的窗口中显示灰度图像
    cv2.imshow('frame', gray)

    # 10 在名为 "mag_spec_shift" 的窗口中显示灰度图像
    cv2.imshow('mag_spec_shift', mag_spec_shift)

    
    # 每隔 1ms 检查一次用户输入，如果按下 'q' 键，退出循环
    if cv2.waitKey(24) == ord('q'):      # 改为24ms,如果是1ms，则播放速度过快。
        break

# 释放视频捕捉资源
cap.release()
# 关闭所有的 GUI 窗口
cv2.destroyAllWindows()