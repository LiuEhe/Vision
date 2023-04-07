import cv2
import numpy as np
from matplotlib import pyplot as plt

def Lzip(mag_spec_sorted,gray, keep):
    # 计算保留振幅的阈值
    thresh = mag_spec_sorted[int((1 - keep) * gray.size)]  # 计算位置的振幅阈值

    # 创建掩码，保留振幅大于阈值的频率分量
    mask = mag_spec > thresh

    # 使用掩码压缩傅里叶系数（模拟通过网络传输压缩后的系数）
    fCoef_compressed = fCoef * mask

    # 使用压缩后的傅里叶系数进行逆傅里叶变换，得到压缩后的图像
    gray_cp = np.fft.ifft2(fCoef_compressed).real

    return gray_cp


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


    # 3 获得幅度谱(振幅谱）并排序
    mag_spec = np.abs(fCoef)
    mag_spec_sorted = np.sort(mag_spec.ravel())  # 对振幅谱进行排序
    
    #4.保留50%的振幅
    gray_1 = Lzip(mag_spec_sorted,gray, 0.5)
    gray_1 = cv2.cvtColor(np.uint8(gray_1), cv2.COLOR_GRAY2BGR) # type: ignore

    # 5.保留10%的振幅
    gray_2 = Lzip(mag_spec_sorted,gray, 0.1)
    gray_2 = cv2.cvtColor(np.uint8(gray_2), cv2.COLOR_GRAY2BGR) # type: ignore


    # 4.显示原视频及其对应的灰度图
    cv2.imshow('original', gray)
    cv2.imshow('0.5', gray_1)
    cv2.imshow('0.1', gray_2)

    
    # 每隔 1ms 检查一次用户输入，如果按下 'q' 键，退出循环
    if cv2.waitKey(24) == ord('q'):      # 改为24ms,如果是1ms，则播放速度过快。
        break

# 释放视频捕捉资源
cap.release()
# 关闭所有的 GUI 窗口
cv2.destroyAllWindows()