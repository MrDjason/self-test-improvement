# ==================== 导入模块 ====================
import cv2
import os
import numpy
# ==================== 主 程 序 ====================
'''1、基础运算'''
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'res')

image_1 = cv2.imread(os.path.join(file_path, 'picture_1.png'))
image_2 = cv2.imread(os.path.join(file_path, 'picture_2.png'))

# 图像加法
result = cv2.add(image_1, image_2)
'''
对于每个像素的每个通道（B、G、R），将两张图对应位置的像素值相加
若相加结果超过 255（像素值的上限），则直接取 255（避免溢出）
'''

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像减法
result = cv2.subtract(image_1, image_2)
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像乘法
result = cv2.multiply(image_1, image_2)
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像除法
result = cv2.divide(image_1, image_2)
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''2.图像位运算'''
# 与运算
result = cv2.bitwise_and(image_1, image_2)
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 或预算
result = cv2.bitwise_or(image_1, image_2)
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 非运算
result = cv2.bitwise_not(image_1)
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 异或运算 
result = cv2.bitwise_xor(image_1, image_2)
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''3.图像混合'''
alpha = 0.7  # 第一幅图像的权重
beta = 0.3   # 第二幅图像的权重
gamma = 0    # 可选的标量值
result = cv2.addWeighted(image_1, alpha, image_2, beta, gamma)
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()