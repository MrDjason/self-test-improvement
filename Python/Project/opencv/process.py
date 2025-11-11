import cv2
from ultralytics import YOLO
import numpy as np

# 加载YOLOv8模型（nano版本，轻量化，适合实时检测）
model = YOLO('yolov8n.pt') 

# 打开摄像头（0为默认摄像头）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 初始化变量：记录上一帧人体检测框的面积（用于判断远近变化）
prev_area = 0
# 阈值：面积变化超过该值才判定为"拉近"或"拉远"
change_threshold = 500  # 可根据实际场景调整

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧，退出")
        break

    # 获取画面尺寸
    frame_h, frame_w = frame.shape[:2]

    # 用YOLO检测画面中的目标
    results = model(frame, classes=[0])  # classes=[0]指定只检测人

    # 提取最大的人体检测框（默认跟踪画面中最明显的人）
    max_area = 0
    best_box = None
    for result in results:
        for box in result.boxes:
            # 检测框坐标：x1,y1（左上角），x2,y2（右下角）
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 计算检测框面积
            area = (x2 - x1) * (y2 - y1)
            # 保留最大的检测框（假设最大的是最近的人）
            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)

    # 处理检测结果
    if best_box is not None:
        x1, y1, x2, y2 = best_box
        # 绘制检测框（绿色）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 标注"Person"
        cv2.putText(frame, "Person", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 判断远近变化（与上一帧比较）
        if prev_area != 0:  # 跳过第一帧（无历史数据）
            area_diff = max_area - prev_area
            # 根据面积变化判断：增大=靠近，减小=远离
            if area_diff > change_threshold:
                cv2.putText(frame, "拉近", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            elif area_diff < -change_threshold:
                cv2.putText(frame, "拉远", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        # 更新上一帧的面积
        prev_area = max_area
    else:
        # 未检测到人时，重置历史面积
        prev_area = 0
        cv2.putText(frame, "未检测到人", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 显示画面
    cv2.imshow('人体距离检测', frame)

    # 按ESC键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()


'''
# ==================== 导入模块 ====================
import cv2
from ultralytics import YOLO
import numpy
# ==================== 主 程 序 ====================
prev_area = 0
change_threshold = 500

def main():
    # 加载YOLO8模型
    model = YOLO('yolov8n.pt')
    #打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap:
        print('摄像头无法打开')
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('错误:无法读取帧')
            break
        frame_h, frame_w = frame[:2] # 获取画面高、宽
        # 用YOLO检测画面
        results = model(frame, classes=[0]) # classes=0只指定人
        for result in results:
            for box in result.boxes:
                # 提取检测框坐标，box.xyxy 左上和右下xy坐标
                x1,y1,x2,y2 = map(int, box.xyxy[0]) 
                # 绘制检测框：绿色
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                # 检测框上方标注"Person"文字
                cv2.putText(frame, "Person", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
                
        # 画面显示和退出
        cv2.imshow('人体检测', frame)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break


# ==================== 运    行 ====================
if __name__ == '__main__':
    main()
'''