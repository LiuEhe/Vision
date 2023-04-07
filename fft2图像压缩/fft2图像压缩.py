from skimage import io, color
import numpy as np
from matplotlib import pyplot as plt
import cv2

#成功显示中文
import matplotlib
matplotlib.rc("font",family='YouYuan')

# 读取灰度图像
img = cv2.imread("lei.jpg", 0)

# 计算图像的二维快速傅里叶变换，得到振幅谱
fCoef = np.fft.fft2(img)
mag_spec = np.abs(fCoef)
mag_spec_sorted = np.sort(mag_spec.ravel())  # 对振幅谱进行排序

# 定义压缩比例列表
compression_ratios = [0.05, 0.01, 0.005, 0.001]

# 初始化画布并显示原始图像
fg, ax = plt.subplots(1, 5, figsize=(20, 10))
ax[0].imshow(img, cmap='gray')
ax[0].set_title("原图")

# 循环处理不同压缩比例
for i, keep in enumerate(compression_ratios):
    # 计算保留振幅的阈值
    thresh = mag_spec_sorted[int((1 - keep) * img.size)]  # 计算位置的振幅阈值
    print((1 - keep))

    # 创建掩码，保留振幅大于阈值的频率分量
    mask = mag_spec > thresh

    # 使用掩码压缩傅里叶系数（模拟通过网络传输压缩后的系数）
    fCoef_compressed = fCoef * mask

    # 使用压缩后的傅里叶系数进行逆傅里叶变换，得到压缩后的图像
    img_cp = np.fft.ifft2(fCoef_compressed).real

    # 将灰度图像转换为二值图像（用于指纹图像，只关注黑白信息，不关注灰度细节）
    img_cp = img_cp > 130

    # 显示压缩后的图像
    ax[i + 1].imshow(img_cp, cmap='gray')
    ax[i + 1].set_title(f"压缩比例：{keep}")

plt.tight_layout()
plt.show()