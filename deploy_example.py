#!/usr/bin/env python3
"""
YOLO 模型部署示例
支持多种部署方式
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
from pathlib import Path

class YOLODeployer:
    """YOLO模型部署器"""
    
    def __init__(self, model_path, device='cuda', conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化部署器
        
        Args:
            model_path: 模型路径 (.pt, .onnx, .engine)
            device: 推理设备
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 获取类别名称
        self.class_names = self.model.names
        
        print(f"模型加载完成!")
        print(f"设备: {device}")
        print(f"类别数量: {len(self.class_names)}")
        
    def inference_image(self, image_path, save_result=True):
        """
        单张图像推理
        
        Args:
            image_path: 图像路径
            save_result: 是否保存结果
            
        Returns:
            推理结果
        """
        print(f"推理图像: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 推理
        start_time = time.time()
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        inference_time = time.time() - start_time
        
        # 处理结果
        result_image = image.copy()
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # 添加到检测列表
                    detections.append({
                        'class': self.class_names[cls_id],
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
                    
                    # 绘制边界框
                    label = f"{self.class_names[cls_id]} {conf:.2f}"
                    cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(result_image, label, (int(x1), int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"推理完成! 耗时: {inference_time:.3f}秒")
        print(f"检测到 {len(detections)} 个目标")
        
        # 保存结果
        if save_result:
            output_path = f"results/{Path(image_path).stem}_result.jpg"
            Path("results").mkdir(exist_ok=True)
            cv2.imwrite(output_path, result_image)
            print(f"结果保存到: {output_path}")
            
            # 保存检测结果到文本文件
            txt_path = f"results/{Path(image_path).stem}_detections.txt"
            with open(txt_path, 'w') as f:
                for det in detections:
                    f.write(f"{det['class']} {det['confidence']:.4f} {det['bbox'][0]} {det['bbox'][1]} {det['bbox'][2]} {det['bbox'][3]}\n")
            print(f"检测结果保存到: {txt_path}")
        
        return {
            'image': result_image,
            'detections': detections,
            'inference_time': inference_time
        }
    
    def inference_video(self, video_path, output_path=None, show_preview=False):
        """
        视频推理
        
        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            show_preview: 是否显示预览
            
        Returns:
            处理统计信息
        """
        print(f"处理视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return None
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
        
        # 准备输出视频
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        total_inference_time = 0
        
        print("开始处理视频...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 推理
            start_time = time.time()
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # 处理结果
            result_frame = frame.copy()
            frame_detections = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    frame_detections = len(boxes)
                    total_detections += frame_detections
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # 绘制边界框
                        label = f"{self.class_names[cls_id]} {conf:.2f}"
                        cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(result_frame, label, (int(x1), int(y1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 添加帧信息
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {frame_detections} | FPS: {1/inference_time:.1f}"
            cv2.putText(result_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 保存或显示
            if output_path:
                out.write(result_frame)
            
            if show_preview:
                cv2.imshow('YOLO Inference', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 进度显示
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧")
        
        # 清理
        cap.release()
        if output_path:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # 统计信息
        avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
        avg_detections = total_detections / frame_count if frame_count > 0 else 0
        
        stats = {
            'total_frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections,
            'total_inference_time': total_inference_time,
            'avg_fps': avg_fps,
            'output_path': output_path
        }
        
        print("\n视频处理完成!")
        print(f"总帧数: {frame_count}")
        print(f"总检测数: {total_detections}")
        print(f"平均每帧检测数: {avg_detections:.2f}")
        print(f"总推理时间: {total_inference_time:.2f}秒")
        print(f"平均FPS: {avg_fps:.2f}")
        if output_path:
            print(f"输出视频: {output_path}")
        
        return stats
    
    def inference_webcam(self, camera_id=0, show_preview=True):
        """
        摄像头实时推理
        
        Args:
            camera_id: 摄像头ID
            show_preview: 是否显示预览
        """
        print(f"启动摄像头 {camera_id} 实时推理...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return
        
        print("按 'q' 键退出")
        
        frame_count = 0
        fps_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            frame_count += 1
            
            # 推理
            start_time = time.time()
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            inference_time = time.time() - start_time
            current_fps = 1 / inference_time if inference_time > 0 else 0
            fps_history.append(current_fps)
            
            # 处理结果
            result_frame = frame.copy()
            detections = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    detections = len(boxes)
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # 绘制边界框
                        label = f"{self.class_names[cls_id]} {conf:.2f}"
                        cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(result_frame, label, (int(x1), int(y1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 添加统计信息
            avg_fps = np.mean(fps_history[-30:]) if len(fps_history) > 0 else 0
            
            info_text = f"FPS: {current_fps:.1f} (Avg: {avg_fps:.1f}) | Detections: {detections}"
            cv2.putText(result_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 显示
            if show_preview:
                cv2.imshow('YOLO Real-time Inference', result_frame)
            
            # 按键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # 清理
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"摄像头推理结束，共处理 {frame_count} 帧")
        print(f"平均FPS: {np.mean(fps_history) if fps_history else 0:.2f}")

def main():
    parser = argparse.ArgumentParser(description='YOLO 模型部署脚本')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径 (.pt, .onnx, .engine)')
    parser.add_argument('--mode', type=str, default='image',
                       choices=['image', 'video', 'webcam'],
                       help='推理模式: image, video, webcam')
    parser.add_argument('--input', type=str,
                       help='输入文件路径 (图像或视频)')
    parser.add_argument('--output', type=str,
                       help='输出文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='推理设备: cuda, cpu, 0, 1')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU阈值')
    parser.add_argument('--show', action='store_true',
                       help='显示预览窗口')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='摄像头ID (webcam模式使用)')
    
    args = parser.parse_args()
    
    # 创建部署器
    deployer = YOLODeployer(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # 根据模式执行推理
    if args.mode == 'image':
        if not args.input:
            print("图像模式需要 --input 参数")
            return
        
        deployer.inference_image(args.input, save_result=True)
        
    elif args.mode == 'video':
        if not args.input:
            print("视频模式需要 --input 参数")
            return
        
        deployer.inference_video(
            video_path=args.input,
            output_path=args.output,
            show_preview=args.show
        )
        
    elif args.mode == 'webcam':
        deployer.inference_webcam(
            camera_id=args.camera_id,
            show_preview=args.show
        )

if __name__ == '__main__':
    main()