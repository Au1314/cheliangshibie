import cv2
import numpy as np
import os
import gc
import shutil
import argparse
from tqdm import tqdm

class Vehicle:
    def __init__(self, vehicle_id, bbox, frame):
        self.id = vehicle_id
        self.plate_number = None
        self.plate_confidence = 0.0
        self.bboxes = [bbox]
        self.frames = [frame]
        self.is_static = False
        self.static_frame_count = 0
        self.last_save_frame = 0
        self.folder_name = None # 记录当前车辆存放的文件夹名
        
    def update_bbox(self, bbox, frame):
        self.bboxes.append(bbox)
        self.frames.append(frame)
        
        # 保持最近10帧的数据
        if len(self.bboxes) > 10:
            self.bboxes = self.bboxes[-10:]
            self.frames = self.frames[-10:]
        
    def mark_static(self):
        self.is_static = True

class IOU_Tracker:
    def __init__(self, iou_threshold=0.3, static_threshold=15, static_frames=10):
        self.vehicles = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.static_threshold = static_threshold
        self.static_frames = static_frames
        self.track_history = {}
        
    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def track(self, detections, frame):
        new_vehicles = []
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            matched = False
            
            for vehicle in self.vehicles:
                last_bbox = vehicle.bboxes[-1]
                iou_score = self.iou((x1, y1, x2, y2), last_bbox)
                
                if iou_score >= self.iou_threshold:
                    vehicle.update_bbox((x1, y1, x2, y2), frame)
                    new_vehicles.append(vehicle)
                    matched = True
                    break
            
            if not matched:
                new_vehicle = Vehicle(self.next_id, (x1, y1, x2, y2), frame.copy())
                self.track_history[self.next_id] = []
                self.next_id += 1
                new_vehicles.append(new_vehicle)
        
        for vehicle in new_vehicles:
            if len(vehicle.bboxes) >= 2:
                prev_center = ((vehicle.bboxes[-2][0] + vehicle.bboxes[-2][2]) // 2, 
                               (vehicle.bboxes[-2][1] + vehicle.bboxes[-2][3]) // 2)
                curr_center = ((vehicle.bboxes[-1][0] + vehicle.bboxes[-1][2]) // 2, 
                               (vehicle.bboxes[-1][1] + vehicle.bboxes[-1][3]) // 2)
                
                distance = np.sqrt((prev_center[0] - curr_center[0])**2 + 
                                  (prev_center[1] - curr_center[1])**2)
                self.track_history[vehicle.id].append(distance)
                
                if len(self.track_history[vehicle.id]) >= self.static_frames:
                    avg_distance = np.mean(self.track_history[vehicle.id][-self.static_frames:])
                    # 如果中心点移动距离小于阈值，认为是静态停放车辆
                    if avg_distance < self.static_threshold and not vehicle.is_static:
                        vehicle.mark_static()
        
        self.vehicles = new_vehicles
        return self.vehicles

class VehicleAnalyzer:
    def __init__(self, output_dir="vehicle_output"):
        self.output_dir = output_dir
        self.tracker = IOU_Tracker()
        self.unknown_counter = 0
        self.output_video_path = ""
        self.known_plates = set() # 记录已经识别出的确切车牌，用于多角度合并
    
    def try_recognize_plate(self, vehicle, lpr):
        """尝试识别车牌，如果识别成功且置信度高，则更新车辆信息"""
        if not lpr: return False
        
        bbox = vehicle.bboxes[-1]
        x1, y1, x2, y2 = bbox
        
        # 确保 bbox 尺寸合理才进行识别，节省 CPU
        if (x2 - x1) > 60 and (y2 - y1) > 60:
            vehicle_crop = vehicle.frames[-1][y1:y2, x1:x2]
            if vehicle_crop.size > 0:
                try:
                    results = lpr.recognize(vehicle_crop)
                    if results and len(results) > 0:
                        plate_str, conf, _ = results[0]
                        # 只有置信度大于 0.8 才认为是有效车牌
                        if conf > 0.8:
                            vehicle.plate_number = plate_str
                            vehicle.plate_confidence = conf
                            return True
                except Exception as e:
                    pass
        return False

    def handle_vehicle_folder(self, vehicle):
        """管理车辆存放的文件夹，处理重命名和多角度合并逻辑"""
        # 1. 首次分配文件夹
        if vehicle.folder_name is None:
            if vehicle.plate_number:
                vehicle.folder_name = vehicle.plate_number
                self.known_plates.add(vehicle.plate_number)
            else:
                self.unknown_counter += 1
                vehicle.folder_name = f"未知车辆_{self.unknown_counter}"
            
            folder_path = os.path.join(self.output_dir, vehicle.folder_name)
            os.makedirs(folder_path, exist_ok=True)
            return folder_path

        # 2. 之前是未知车辆，现在识别出了车牌 (多角度合并的核心逻辑)
        if vehicle.folder_name.startswith("未知车辆_") and vehicle.plate_number:
            old_folder_path = os.path.join(self.output_dir, vehicle.folder_name)
            new_folder_name = vehicle.plate_number
            new_folder_path = os.path.join(self.output_dir, new_folder_name)
            
            if new_folder_name in self.known_plates and os.path.exists(new_folder_path):
                # 目标车牌文件夹已存在 (说明这是同一辆车的不同角度/不同追踪ID)
                # 将未知文件夹里的旧图片全部移动到已有的车牌文件夹中
                if os.path.exists(old_folder_path):
                    for file in os.listdir(old_folder_path):
                        shutil.move(os.path.join(old_folder_path, file), os.path.join(new_folder_path, file))
                    os.rmdir(old_folder_path) # 删除空的未知文件夹
            else:
                # 目标车牌文件夹不存在，直接重命名
                if os.path.exists(old_folder_path):
                    os.rename(old_folder_path, new_folder_path)
                else:
                    os.makedirs(new_folder_path, exist_ok=True)
                self.known_plates.add(new_folder_name)
            
            # 更新该车辆当前的文件夹名
            vehicle.folder_name = new_folder_name
            return new_folder_path
            
        # 3. 正常返回已分配的文件夹
        return os.path.join(self.output_dir, vehicle.folder_name)
    
    def save_vehicle_image(self, vehicle, folder_path, suffix):
        bbox = vehicle.bboxes[-1]
        x1, y1, x2, y2 = bbox
        
        if x2 > x1 and y2 > y1:
            crop = vehicle.frames[-1][y1:y2, x1:x2]
            if crop.size > 0:
                img_path = os.path.join(folder_path, f"{vehicle.id}_{suffix}.jpg")
                cv2.imwrite(img_path, crop)
    
    def process_video(self, video_path):
        os.makedirs(self.output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_video_path = os.path.join(self.output_dir, f"{video_name}_processed.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        print("正在加载 YOLOv8n 模型 (基于CPU)...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            model.to('cpu')
        except Exception as e:
            print(f"YOLO模型加载失败: {str(e)}")
            cap.release()
            out.release()
            return
        
        print("正在加载 HyperLPR3 车牌识别...")
        try:
            from hyperlpr3 import LicensePlateRecognizer
            lpr = LicensePlateRecognizer()
        except Exception as e:
            print(f"警告: 车牌识别模块加载失败: {str(e)}")
            lpr = None
        
        frame_interval = 15  # 保存图片的间隔帧数
        frame_counter = 0
        
        print("正在分析视频...")
        with tqdm(total=total_frames, desc="处理进度") as pbar:
            try:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    frame_counter += 1
                    pbar.update(1)
                    
                    # 4核CPU优化：隔帧处理，减轻CPU负担
                    if frame_counter % 2 != 0:
                        out.write(frame)
                        continue
                    
                    # 仅检测车辆 (classes=[2] 在 COCO 中代表 car)
                    results = model.predict(frame, classes=[2, 3, 5, 7], conf=0.3, verbose=False)
                    
                    detections = []
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            detections.append((x1, y1, x2, y2, conf, cls))
                    
                    vehicles = self.tracker.track(detections, frame)
                    display_frame = frame.copy()
                    
                    for vehicle in vehicles:
                        bbox = vehicle.bboxes[-1]
                        x1, y1, x2, y2 = bbox
                        
                        if vehicle.is_static:
                            color = (0, 255, 0)
                            status = "静态"
                            vehicle.static_frame_count += 1
                            
                            # 如果该静态车辆目前还没有识别出车牌，则每隔几帧尝试重新识别一次
                            if not vehicle.plate_number and vehicle.static_frame_count % 10 == 0:
                                self.try_recognize_plate(vehicle, lpr)
                            
                            # 获取或更新该车辆对应的文件夹路径 (核心：处理多角度和重命名)
                            folder_path = self.handle_vehicle_folder(vehicle)
                            
                            # 保存图像逻辑
                            if vehicle.static_frame_count == 1:
                                self.save_vehicle_image(vehicle, folder_path, "initial")
                                vehicle.last_save_frame = frame_counter
                            elif frame_counter - vehicle.last_save_frame >= frame_interval:
                                self.save_vehicle_image(vehicle, folder_path, f"frame_{frame_counter}")
                                vehicle.last_save_frame = frame_counter
                                
                        else:
                            color = (0, 0, 255)
                            status = "移动"
                        
                        # 绘制框和文字
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        plate_text = vehicle.plate_number if vehicle.plate_number else (vehicle.folder_name if vehicle.folder_name else "未知")
                        text = f"ID:{vehicle.id} {plate_text} {status}"
                        cv2.putText(display_frame, text, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    out.write(display_frame)
                    
                    # 定期清理内存，防止云服务器内存爆满
                    if frame_counter % 100 == 0:
                        gc.collect()
            except Exception as e:
                print(f"处理过程中发生错误: {str(e)}")
            finally:
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                gc.collect()
        
        print(f"\n分析完成！")
        print(f"共检测出 {len(self.known_plates)} 辆确切车牌的静态车辆")
        print(f"处理后视频已保存到: {self.output_video_path}")
        print(f"车辆图片已保存到: {self.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="车辆分析程序 - 静态车辆识别、车牌提取、多角度图像合并")
    parser.add_argument("--input", "-i", required=True, help="输入视频文件路径")
    parser.add_argument("--output", "-o", default="vehicle_output", help="输出目录路径")
    
    args = parser.parse_args()
    
    analyzer = VehicleAnalyzer(output_dir=args.output)
    analyzer.process_video(args.input)