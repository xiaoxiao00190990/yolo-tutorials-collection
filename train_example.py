#!/usr/bin/env python3
"""
YOLOv8 快速训练示例
适用于自定义数据集训练
"""

import os
import yaml
from ultralytics import YOLO
import argparse

def prepare_data_config(data_dir, class_names):
    """
    准备数据集配置文件
    """
    config = {
        'path': os.path.abspath(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(data_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'labels', split), exist_ok=True)
    
    # 保存配置文件
    config_path = os.path.join(data_dir, 'data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"配置文件已创建: {config_path}")
    return config_path

def train_model(data_yaml, model_size='yolov8n', epochs=100, imgsz=640, device='cuda'):
    """
    训练YOLO模型
    """
    print(f"开始训练 {model_size} 模型...")
    print(f"数据集: {data_yaml}")
    print(f"训练轮数: {epochs}")
    print(f"图像大小: {imgsz}")
    print(f"设备: {device}")
    
    # 加载模型
    if model_size.endswith('.pt'):
        model = YOLO(model_size)
    else:
        model = YOLO(f'{model_size}.pt')
    
    # 训练参数
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': 16,
        'device': device,
        'workers': 4,
        'save': True,
        'save_period': 10,
        'project': 'runs/train',
        'name': 'exp',
        'exist_ok': True,
        'verbose': True
    }
    
    # 开始训练
    results = model.train(**train_args)
    
    print("训练完成!")
    print(f"最佳模型保存在: runs/train/exp/weights/best.pt")
    
    return results

def export_model(model_path, format='onnx'):
    """
    导出模型到不同格式
    """
    print(f"导出模型到 {format} 格式...")
    
    model = YOLO(model_path)
    
    if format == 'onnx':
        success = model.export(format='onnx', simplify=True)
    elif format == 'torchscript':
        success = model.export(format='torchscript')
    elif format == 'tflite':
        success = model.export(format='tflite')
    elif format == 'coreml':
        success = model.export(format='coreml')
    else:
        print(f"不支持的格式: {format}")
        return False
    
    if success:
        print(f"模型已导出到: {model_path.replace('.pt', f'.{format}')}")
        return True
    else:
        print("模型导出失败")
        return False

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 训练脚本')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'export', 'prepare'],
                       help='运行模式: train, export, prepare')
    parser.add_argument('--data-dir', type=str, default='./datasets/custom',
                       help='数据集目录')
    parser.add_argument('--classes', type=str, nargs='+',
                       default=['object1', 'object2'],
                       help='类别名称列表')
    parser.add_argument('--model', type=str, default='yolov8n',
                       help='模型类型: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备: cuda, cpu, 0, 1')
    parser.add_argument('--weights', type=str,
                       default='runs/train/exp/weights/best.pt',
                       help='模型权重路径 (用于导出模式)')
    parser.add_argument('--export-format', type=str, default='onnx',
                       choices=['onnx', 'torchscript', 'tflite', 'coreml'],
                       help='导出格式')
    
    args = parser.parse_args()
    
    if args.mode == 'prepare':
        # 准备数据配置
        config_path = prepare_data_config(args.data_dir, args.classes)
        print(f"数据配置准备完成: {config_path}")
        
    elif args.mode == 'train':
        # 训练模型
        data_yaml = os.path.join(args.data_dir, 'data.yaml')
        if not os.path.exists(data_yaml):
            print(f"数据配置文件不存在: {data_yaml}")
            print("请先运行 --mode prepare 创建配置文件")
            return
        
        train_model(data_yaml, args.model, args.epochs, args.imgsz, args.device)
        
    elif args.mode == 'export':
        # 导出模型
        if not os.path.exists(args.weights):
            print(f"模型权重文件不存在: {args.weights}")
            return
        
        export_model(args.weights, args.export_format)

if __name__ == '__main__':
    main()