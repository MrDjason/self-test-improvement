# ==================== 导入模块 ====================
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
# ==================== 全局变量 ====================
change_threshold = 3000      # 小面积触发放大
scale_factor = 2.5           # 放大倍数
min_crop_ratio = 0.2         # 最小裁剪比例
track_threshold = 5          # 丢失重连帧数
smooth_alpha = 0.3           # 平滑参数（0~1之间，越小越稳）
area_buffer_len = 5          # 放大触发平滑窗口
# ==================== 主 程 序 ====================
def smooth_box(prev_box, new_box, alpha=0.3):
    """对检测框中心与尺寸进行平滑"""
    if prev_box is None:
        return new_box
    (x1p, y1p, x2p, y2p) = prev_box
    (x1, y1, x2, y2) = new_box
    return (
        int(x1p * (1 - alpha) + x1 * alpha),
        int(y1p * (1 - alpha) + y1 * alpha),
        int(x2p * (1 - alpha) + x2 * alpha),
        int(y2p * (1 - alpha) + y2 * alpha)
    )

def main():
    # 加载YOLOv8模型
    model = YOLO("yolov8n.pt")
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头无法打开")
        return

    # 初始化追踪相关变量
    track_window = None # CamShift的追踪窗口（x,y,w,h）
    roi_hist = None     # 目标区域的直方图（用于CamShift追踪）
    is_tracking = False # 是否正在追踪的标记
    lost_count = 0      # 目标丢失的帧数计数器
    prev_box = None     # 上一帧的检测框/追踪框（用于平滑）
    # CamShift的终止条件：迭代10次或移动距离小于1时停止
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    area_buffer = deque(maxlen=area_buffer_len) # 缓存目标面积的队列（用于平滑判断）

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_h, frame_w = frame.shape[:2] # 获取画面尺寸
        original_ratio = frame_w / frame_h # 计算原始宽高比
        max_box = None # 存储YOLO检测到的最大人体框
        max_area = 0   # 存储最大人体框的面积

        # YOLO 检测（人体）
        results = model(frame, classes=[0], verbose=False) # 调用YOLO检测，关闭输出日志
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area: # 保留面积最大的框（最显著的人体目标）
                    max_area = area
                    max_box = (x1, y1, x2, y2)

        # 框平滑
        if max_box is not None: # 如果检测到目标
            # 用前一帧的框和当前框做平滑处理
            max_box = smooth_box(prev_box, max_box, smooth_alpha)
            prev_box = max_box # 更新前一帧的框为当前平滑后的框

        # 初始化/更新 CamShift
        if max_box is not None:             # 如果检测到目标
            x1, y1, x2, y2 = max_box        # 解包当前框的坐标
            w, h = x2 - x1, y2 - y1         # 计算框的宽和高
            current_window = (x1, y1, w, h) # CamShift需要的窗口格式（x,y,w,h）

            # 如果未追踪或目标丢失超过阈值，重新初始化CamShift
            if not is_tracking or lost_count > track_threshold:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                track_window = current_window
                is_tracking = True
                lost_count = 0
        else: # 如果未检测到目标
            lost_count += 1 # 增加丢失计数
            if lost_count > track_threshold: # 如果丢失超过阈值，停止追踪
                is_tracking = False
                roi_hist = None

        # CamShift 追踪
        tracked_box = None  # 存储CamShift追踪到的框
        tracked_area = 0    # 存储追踪框的面积
        camshift_ret = None # 存储CamShift的返回结果（旋转矩形）

        if is_tracking and roi_hist is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            camshift_ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            x, y, w, h = track_window
            tracked_box = (x, y, x + w, y + h)
            tracked_area = w * h

            # 平滑追踪框
            tracked_box = smooth_box(prev_box, tracked_box, alpha=0.2)
            prev_box = tracked_box

        # 放大逻辑（使用面积移动平均）
        area_buffer.append(tracked_area)
        avg_area = np.mean(area_buffer) if area_buffer else tracked_area

        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, frame_w, frame_h

        if tracked_box is not None:
            x1_t, y1_t, x2_t, y2_t = tracked_box
            center_x = (x1_t + x2_t) / 2
            center_y = (y1_t + y2_t) / 2

            # 平滑面积判断
            if avg_area > change_threshold:
                crop_w = frame_w / scale_factor
                crop_h = crop_w / original_ratio
                min_crop_w = frame_w * min_crop_ratio
                min_crop_h = frame_h * min_crop_ratio
                crop_w = max(crop_w, min_crop_w)
                crop_h = max(crop_h, min_crop_h)

                crop_x1 = int(max(0, center_x - crop_w / 2))
                crop_y1 = int(max(0, center_y - crop_h / 2))
                crop_x2 = int(min(frame_w, crop_x1 + crop_w))
                crop_y2 = int(min(frame_h, crop_y1 + crop_h))

        # 裁剪 + 放大显示
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        # INTER_LANCZOS4：放大时画质更好
        display_frame = cv2.resize(cropped_frame, (frame_w, frame_h), interpolation=cv2.INTER_LANCZOS4)

        # 绘制追踪框
        if is_tracking and camshift_ret is not None:
            pts = cv2.boxPoints(camshift_ret)
            pts = np.int32(pts)
            # 计算裁剪区域的原始尺寸
            crop_original_w = crop_x2 - crop_x1
            crop_original_h = crop_y2 - crop_y1

            if crop_original_w > 0 and crop_original_h > 0:
            # 将原始帧中的追踪框坐标转换为显示帧的坐标
                pts_scaled = [(int((px - crop_x1)/crop_original_w*frame_w),
                               int((py - crop_y1)/crop_original_h*frame_h)) for (px, py) in pts]
                pts_scaled = np.int32(pts_scaled)
                # 绘制旋转矩形（追踪框）
                cv2.polylines(display_frame, [pts_scaled], True, (0, 0, 255), 2)
                # 添加"Tracking"文本
                cv2.putText(display_frame, "Tracking", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示画面
        cv2.imshow("稳定远距离人追踪放大", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 退出条件
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
