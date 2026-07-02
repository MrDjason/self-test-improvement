# ==================== 导入模块 ====================
import cv2
import os
# ==================== 主 程 序 ====================
# 读取文件
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'res')

image_path = file_path + '/bird.jpg'
image = cv2.imread(image_path)

# 检测是否成功读取
if image is None:
    print('错误：无法加载图像，请检查路径是否正确')
    exit()

# 显示图像
cv2.imshow('Display Image', image) # 创建一个名为Display Image的窗口，并在其中显示图像

# 等待用户按键
key = cv2.waitKey(0) # 参数0代表无限等待 直到用户按下任意按键

# 根据用户案件执行操作
if key == ord('s'): # 如果按下's'键
    output_path = file_path + '/save_image.jpg'
    cv2.imwrite(output_path, image)
    print(f'图像已保存{output_path}')
else:
    print('图像未保存')

# 关闭所有窗口
cv2.destroyAllWindows()