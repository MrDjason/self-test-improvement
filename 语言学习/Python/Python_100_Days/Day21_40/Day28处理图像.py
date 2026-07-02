# Day 28 处理图像
# ==================== 导入模块 ====================
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import os
import random

# ==================== 主 程 序 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'res', 'guido.jpg')
image = Image.open(file_path) # 读取图像获得Image对象
print(image.format)  # 通过format获取图像的格式
print(image.size)    # 通过size获取图像尺寸
print(image.mode)    # 通过show显示图像
image.show()

# 裁剪
image.crop((80, 20, 310, 360)).show() # 通过crop方法指定区域裁剪图像

# 缩略图
image.thumbnail((128, 128))
image.show()

# 缩放和复制
user_path = os.path.join(script_dir, 'res', 'user.png')
user = Image.open(user_path)
guido_image = Image.open(file_path)
guido_head = guido_image.crop((80, 20 ,310, 360))
width, height = guido_head.size
width, height = guido_head.size
resized_head = guido_head.resize((int(width * 4), int(height * 4)))

paste_x = (user.width - resized_head.width) // 2
paste_y = 30
user.paste(resized_head, (paste_x, paste_y))
user.show()

# 翻转
image = Image.open(file_path)
# 使用rotate方法实现图像旋转
image.rotate(45).show()
# 使用transpose方法实现图像翻转
# Image.FLIP_LEFT_RIGHT - 水平翻转
# Image.FLIP_TOP_BOTTOM - 垂直翻转
image.transpose(Image.FLIP_TOP_BOTTOM).show()

# 操作像素
for x in range(80, 310):
    for y in range(20, 360):
        image.putpixel((x,y), (128, 128, 128))
image.show()

image.filter(ImageFilter.CONTOUR).show()


# 绘制图片
def random_color():
    '''生成随机颜色'''
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0,255)
    return red, green, blue

width, height = 800, 600
image = Image.new(mode='RGB', size=(width, height), color=(255,255,255))
drawer = ImageDraw.Draw(image)
font = ImageFont.truetype(r'C:\Windows\Fonts\simhei.ttf', 32)  # 通过指定字体和大小获得ImageFont对象
drawer.text((300, 50), 'Hello, world!', fill=(255, 0, 0), font=font) # 通过text方法绘制文字
drawer.line((0, 0, width, height), fill=(0, 0, 255), width=2) # 通过line方法绘制两条对角直线
drawer.line((width, 0, 0, height), fill=(0, 0, 255), width=2)
xy = width // 2 - 60, height // 2 - 60, width // 2 + 60, height // 2 + 60
# 通过ImageDraw对象的rectangle方法绘制矩形
drawer.rectangle(xy, outline=(255, 0, 0), width=2)
# 通过ImageDraw对象的ellipse方法绘制椭圆
for i in range(4):
    left, top, right, bottom = 150 + i * 120, 220, 310 + i * 120, 380
    drawer.ellipse((left, top, right, bottom), outline=random_color(), width=8)
# 显示图像
image.show()
# 保存图像
image.save(script_dir + '/res' + '/result.png')