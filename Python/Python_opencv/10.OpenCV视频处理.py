# ==================== 导入模块 ====================
import cv2
import os
# ==================== 主 程 序 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'res')

'''读取视频文件'''
cap = cv2.VideoCapture(os.path.join(file_path, 'example.mp4'))

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 读取视频帧
while True:
    ret, frame = cap.read()
   
    # 如果读取到最后一帧，退出循环
    if not ret:
        break
   
    # 显示当前帧
    cv2.imshow('Video', frame)
   
    # 按下 'q' 键退出
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

'''读取摄像头视频'''
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('错误：摄像头没有打开')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Camera', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

'''视频帧处理'''
cap = cv2.VideoCapture(os.path.join(file_path, 'example.mp4'))
while True:
    ret, frame=cap.read()

    if not ret:
        break
    # 将帧转为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 显示灰度帧
    cv2.imshow('Gray Video', gray_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

'''帧的保存'''
# 获取视频的帧率和尺寸
cap = cv2.VideoCapture(os.path.join(file_path, 'example.mp4'))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象，保存处理后的视频
fourcc = cv2.VideoWriter_fourcc(*'XVID') # 指定视频的编码格式为XVID，一种常用的MPEG-4视频编码器
out = cv2.VideoWriter(os.path.join(file_path, 'output.avi'), fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    # 将帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 将灰度帧写入输出视频
    out.write(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR))
    # 显示灰度帧
    cv2.imshow('Gray Video', gray_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

'''视频处理高级应用'''
# 加载 Haar 特征分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # 在帧上绘制矩形框标记人脸
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # x,y为左上角顶点位置，w,h为宽高，颜色，厚度
    
    # 显示带有人脸标记的帧
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''运动检测'''
cap = cv2.VideoCapture(0)

# 读取第一帧
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
   
    if not ret:
        break
   
    # 将当前帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # 计算当前帧与前一帧的差异
    frame_diff = cv2.absdiff(prev_gray, gray_frame)
   
    # 对差异图像进行二值化处理
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
   
    # 显示运动检测结果
    cv2.imshow('Motion Detection', thresh)
   
    # 更新前一帧
    prev_gray = gray_frame
   
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()