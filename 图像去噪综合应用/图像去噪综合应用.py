# skimage.restoration 提供了图像去噪的方法，这里我们使用 denoise_wavelet 方法进行小波去噪
# skimage.io 提供了图像的输入输出功能，这里我们使用 io.imread 读取图像
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from skimage.io import imread
from skimage import io

from skimage import img_as_float, io
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means)
import numpy as np

#读图并添加噪音
img = io.imread("./img/cat.jpg", as_gray=True) #灰度图
cat_noise=random_noise(img)  #噪音图

# 利用小波去噪
# 使用 denoise_wavelet 方法对含有噪音的图像 noisy_img 进行小波去噪
start = np.linspace(0,1,101)
start = start[1:]

X = []
Y = []
#print(start)
for i in start:
    wavelet_img = denoise_wavelet(cat_noise,sigma=i)
    noise_psnr = psnr(img, wavelet_img)
    X.append(i)
    Y.append(noise_psnr)

#转化成np数组
X = np.array(X)
Y = np.array(Y)
#print(Y)

#找出最佳效果
sigma_best = np.argsort(Y)[-1]
wavelet_img_best = denoise_wavelet(cat_noise,sigma=(sigma_best/100))

# 使用 matplotlib 绘图
# 创建 1 行 4 列的子图，figsize 设置图像的尺寸
fg, ax = plt.subplots(1, 4, figsize=(20, 2))

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Noise-Free Image')  # 设置子图标题


# # 在第二个子图中展示原始的噪音图像
ax[1].imshow(cat_noise, cmap='gray')
ax[1].set_title('Noisy Image')  # 设置子图标题

# # 在第三个子图中展示小波去噪后的结果
ax[2].imshow(wavelet_img_best, cmap='gray')
ax[2].set_title(f'Wavelet Denoised Best Sigma:0.0{sigma_best},PSNR:{round(Y.max(),2)}')  # 设置子图标题

# 画出峰噪比图
ax[3].plot(X,Y)
ax[3].set_title("PSNR vs Sigma")
#print(X.shape,Y.shape)

ax[3].annotate(f'({sigma_best/100},{round(Y.max(),2)})',    #名字
             xy=(sigma_best/100,Y.max()),xytext=(0.8,26),weight='bold',color='r',   #指向點
             
             arrowprops = dict(arrowstyle="->",color='r',   #箭頭格式
                               connectionstyle="angle3")   #箭頭綫改成彎曲
            )

# 显示图像
plt.show()