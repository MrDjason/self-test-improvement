# ==================== 基本信息 ====================
'''
基本模块
cv2.core       包含了图像处理基本功能(数组表示和操作)
cv2.imgproc    图像处理模块，提供图像各种操作——滤波、图像变换、形态学操作
cv2.highgui    图形用户的界面模块，提供显示图像和视频功能
cv2.video      提供视频处理的共呢个，如视频捕捉、视频流处理等
cv2.features2d 特征检测与匹配模块，包括角点、边缘、关键点检测等
cv2.ml         机器学习模块，提供多种机器学习算法，可进行图像分类、回归、聚类等
cv2.calib3d    相机校准和3D重建模块
cv2.objdetect  目标检测模块
cv2.dnn        深度学习模块
'''
# ==================== 导入模块 ====================
import cv2
import os
import numpy
# ==================== 主 程 序 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'res')


# 显示图像和基本操作
image_path = os.path.join(file_path, 'image.jpg')
image = cv2.imread(image_path)                        # 显示图像
cv2.imshow('Display Window', image)      
key = cv2.waitKey(0)                                  # 等待输入，显示图像时必须添加

if key == ord('x'):       
    # 图像基本操作        
    pix_value = image[100,100]                        # 获取(100, 100)处像素值
    image[100, 100] = [255, 255, 255]                 # 修改像素值为白色
    roi = image[50:150, 50:150]                       # 获取roi(感兴趣区域)
    image[50:150, 50:150] = [0, 255, 0]               # 将roi区域设置为绿色
    cv2.imwrite(file_path+'/output_image.jpg', image) # 保存图片
    print('保存成功')
else:                                   # waitKey返回ascii码，用ord返回字符对应ascii码
    cv2.destroyAllWindows()                           # 关闭窗口
    print('成功退出')

image = cv2.imread(file_path+'/image1.jpg')  

# 图像通道分离
b, g, r = cv2.split(image) # 将图像拆分成bgr三个单通道灰度图
cv2.imshow('Display Window', b)      
key = cv2.waitKey(0)
cv2.imshow('Display Window', g)      
key = cv2.waitKey(0)          
cv2.imshow('Display Window', r)      
key = cv2.waitKey(0)                          
cv2.destroyAllWindows() 

# 图像通道合并
merged_image = cv2.merge([b, g, r])

# 图像缩放、旋转、平移、反转
resized_image = cv2.resize(image, dsize=(1500, 1500)) 
cv2.imshow('resize', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 从图像的形状信息中提取高度（h）和宽度（w）
(h,w) = image.shape[:2] 
# image.shape对彩色图像返回一个如(480, 640, 3)的元组，对灰度图像返回(480, 640)
center = (w//2, h//2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0) # 生成旋转矩阵，定义旋转变换规则：旋转中心，角度，放缩
rotated_image = cv2.warpAffine(image, rotation_matrix,(1500, 1500)) # 对图像进行仿射变换
cv2.imshow('rotation', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

translation_matrix = numpy.float32([[1, 0, 100], [0, 1, 50]])  # 定义平移矩阵
# [1, 0, 100] 100表示向右平移100像素，-100表示向左平移100个像素
# 10 01这几个值保证矩阵平移不缩放
translated_image = cv2.warpAffine(image, translation_matrix, (w,h))
cv2.imshow('translated', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 翻转
flipped_image = cv2.flip(image, 1)  # 第二个位置: 0 (垂直翻转), 1 (水平翻转), -1 (双向翻转)
cv2.imshow('flipped', flipped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

