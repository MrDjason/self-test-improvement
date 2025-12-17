# ==================== 导入模块 ====================
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os

import numpy as np
import random
# ==================== 导入模块 ====================
def imshow(img_path, transform):
# 展示图象
    '''
    Param img_path:图像路径
    Param transform:要应用的数据增强技术
    '''
    img = Image.open(img_path)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].set_title(f'Original image{img.size}')
    ax[0].imshow(img)
    img = transform(img)
    ax[1].set_title(f'Transformed image {img.size}')
    ax[1].imshow(img)
    plt.show()


def cutout_pil_multi(image, mask_size=50, num_masks=3):
# 遮罩
    """
    对图像应用多个 Cutout 遮挡块

    参数:
    - image: PIL.Image 对象
    - mask_size: 每个遮挡块的大小（正方形边长）
    - num_masks: 遮挡块的数量
    """
    image_np = np.array(image).copy()
    h, w = image_np.shape[0], image_np.shape[1]

    for _ in range(num_masks):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        y1 = max(0, y - mask_size // 2)
        y2 = min(h, y + mask_size // 2)
        x1 = max(0, x - mask_size // 2)
        x2 = min(w, x + mask_size // 2)

        # 遮挡区域设置为黑色
        image_np[y1:y2, x1:x2, :] = 0

    return Image.fromarray(image_np)

if __name__ == '__main__':

    # 水平翻转
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'res', 'PetImages')
    transform = transforms.RandomHorizontalFlip(p=1.0) # p=1.0表示总是反转，p是翻转的概率值
    path = os.path.join(DATA_DIR, 'Cat', '6039.jpg')
    imshow(path, transform)

    # 垂直翻转
    transform = transforms.RandomVerticalFlip(p=1.0)
    imshow(path, transform)

    # 随机裁剪
    transform = transforms.RandomCrop(size=(120,120))
    imshow(path, transform)

    # 透视变换
    transform = transforms.RandomPerspective(
        distortion_scale=0.5,
        p=1.0
        interpolation=transforms.InterpolationMode.BILINEAR
    )
    imshow(path, transform)

    # 颜色变化
    # 亮度、对比度、饱和度、色调
    transforms.ColorJitter(
        brightness = 0.5,  # 亮度
        contrast = 0.5,    # 对比度
        saturation = 0.5,  # 饱和度
        hue = 0.1          # 色调
    )
    imshow(path, transform)

    # 模糊
    # 高斯模糊
    transform = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
    imshow(path, transform)

    # 遮罩
    # 在训练图像上随机遮挡一个或多个连续的方形区域
