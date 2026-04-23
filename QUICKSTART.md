# 🚀 YOLO 快速开始指南

本指南将帮助您快速开始使用YOLO进行目标检测。

## 1. 环境准备

### 1.1 安装Python环境
```bash
# 创建虚拟环境
python -m venv yolo-env
source yolo-env/bin/activate  # Linux/Mac
# 或
yolo-env\Scripts\activate  # Windows

# 升级pip
pip install --upgrade pip
```

### 1.2 安装依赖
```bash
# 安装基础依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装YOLOv8
pip install ultralytics

# 安装其他依赖
pip install opencv-python pillow matplotlib numpy pandas pyyaml
```

### 1.3 验证安装
```python
import torch
from ultralytics import YOLO

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")

# 测试YOLO
model = YOLO('yolov8n.pt')
print("YOLO安装成功!")
```

## 2. 数据准备

### 2.1 数据集结构
```
datasets/
├── custom/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
```

### 2.2 创建data.yaml
```yaml
# data.yaml
path: ./datasets/custom
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 2

# 类别名称
names: ['cat', 'dog']
```

### 2.3 标注格式 (YOLO格式)
```
# label.txt
<class_id> <x_center> <y_center> <width> <height>

# 示例
0 0.5 0.5 0.3 0.4  # 类别0在图像中心，宽30%，高40%
```

## 3. 训练模型

### 3.1 使用命令行
```bash
# 训练YOLOv8n模型
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640

# 使用GPU训练
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 device=0

# 继续训练
yolo train resume model=runs/train/exp/weights/last.pt
```

### 3.2 使用Python脚本
```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 训练
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda',  # 或 'cpu'
    workers=4,
    save=True,
    save_period=10,
    project='runs/train',
    name='exp'
)

print("训练完成!")
```

### 3.3 训练参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| data | 数据集配置文件路径 | - |
| epochs | 训练轮数 | 100 |
| imgsz | 输入图像大小 | 640 |
| batch | 批次大小 | 16 |
| device | 训练设备 | 'cpu' |
| workers | 数据加载线程数 | 8 |
| patience | 早停耐心值 | 100 |
| save | 是否保存模型 | True |
| save_period | 保存间隔(轮数) | -1 |

## 4. 模型评估

### 4.1 评估训练好的模型
```bash
# 评估模型
yolo val model=runs/train/exp/weights/best.pt data=data.yaml

# 在测试集上评估
yolo val model=best.pt data=data.yaml split=test
```

### 4.2 使用Python评估
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/exp/weights/best.pt')

# 评估
metrics = model.val(
    data='data.yaml',
    split='val',  # val, test
    device='cuda'
)

print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")
```

## 5. 模型推理

### 5.1 图像推理
```python
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('best.pt')

# 单张图像推理
results = model('image.jpg')

# 显示结果
results[0].show()

# 保存结果
results[0].save('result.jpg')

# 获取检测信息
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            print(f"类别: {model.names[int(box.cls)]}")
            print(f"置信度: {box.conf:.4f}")
            print(f"边界框: {box.xyxy}")
```

### 5.2 视频推理
```python
from ultralytics import YOLO

model = YOLO('best.pt')

# 视频推理
results = model.predict(
    source='video.mp4',
    conf=0.25,
    save=True,
    show=True
)

# 实时摄像头
results = model.predict(
    source=0,  # 摄像头ID
    show=True,
    stream=True  # 实时流
)
```

### 5.3 批量推理
```bash
# 批量处理图像
yolo predict model=best.pt source='images/*.jpg'

# 处理整个文件夹
yolo predict model=best.pt source='folder/'

# 保存结果到指定目录
yolo predict model=best.pt source='images/' project='results' name='exp'
```

## 6. 模型导出

### 6.1 导出为不同格式
```bash
# 导出为ONNX
yolo export model=best.pt format=onnx

# 导出为TensorRT
yolo export model=best.pt format=engine device=0

# 导出为TorchScript
yolo export model=best.pt format=torchscript

# 导出为CoreML
yolo export model=best.pt format=coreml
```

### 6.2 使用Python导出
```python
from ultralytics import YOLO

model = YOLO('best.pt')

# 导出为ONNX
success = model.export(format='onnx', simplify=True)

# 导出为TensorRT
success = model.export(format='engine', device=0)

if success:
    print("模型导出成功!")
```

## 7. 模型部署

### 7.1 使用导出的模型
```python
from ultralytics import YOLO

# 加载ONNX模型
model = YOLO('best.onnx')

# 加载TensorRT模型
model = YOLO('best.engine')

# 推理方式相同
results = model('image.jpg')
```

### 7.2 创建Web API
```python
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO('best.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 读取图像
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 推理
    results = model(image)
    
    # 处理结果
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                detections.append({
                    'class': model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })
    
    return {'detections': detections}
```

### 7.3 使用Docker部署
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY . .
COPY best.pt ./models/

# 运行API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 8. 性能优化

### 8.1 模型选择
| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| YOLOv8n | 3.2M | ⚡⚡⚡ | ⭐ | 移动端、边缘设备 |
| YOLOv8s | 11.2M | ⚡⚡ | ⭐⭐ | 平衡型 |
| YOLOv8m | 25.9M | ⚡ | ⭐⭐⭐ | 服务器部署 |
| YOLOv8l | 43.7M | 🐢 | ⭐⭐⭐⭐ | 高精度需求 |
| YOLOv8x | 68.2M | 🐢🐢 | ⭐⭐⭐⭐⭐ | 研究、竞赛 |

### 8.2 推理优化技巧
```python
# 1. 使用半精度推理
model = YOLO('best.pt')
results = model('image.jpg', half=True)  # FP16

# 2. 调整置信度阈值
results = model('image.jpg', conf=0.5)  # 提高阈值减少误检

# 3. 批量推理
results = model(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# 4. 使用TensorRT加速
model = YOLO('best.engine')  # TensorRT引擎
```

### 8.3 内存优化
```python
# 清理GPU缓存
import torch
torch.cuda.empty_cache()

# 使用梯度检查点
model.train(checkpoint=True)

# 减少批次大小
model.train(batch=8)
```

## 9. 常见问题

### 9.1 训练问题
**Q: 训练损失不下降**
- 检查学习率是否合适
- 检查数据标注是否正确
- 尝试更小的模型或更多数据

**Q: 过拟合**
- 增加数据增强
- 使用早停
- 添加正则化

### 9.2 推理问题
**Q: 推理速度慢**
- 使用更小的模型
- 启用半精度推理
- 使用TensorRT加速

**Q: 检测精度低**
- 调整置信度阈值
- 检查类别不平衡
- 重新训练或微调

### 9.3 部署问题
**Q: 模型导出失败**
- 检查PyTorch版本
- 确保模型格式正确
- 查看错误日志

**Q: 内存不足**
- 减少批次大小
- 使用CPU推理
- 优化模型大小

## 10. 下一步

### 学习资源
1. [Ultralytics文档](https://docs.ultralytics.com/)
2. [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
3. [PyTorch教程](https://pytorch.org/tutorials/)

### 进阶主题
1. 自定义数据增强
2. 模型蒸馏和剪枝
3. 多任务学习
4. 实时部署优化

### 社区支持
- [Ultralytics Discord](https://discord.gg/ultralytics)
- [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/yolo)

---

**提示**: 本指南基于YOLOv8编写，其他YOLO版本可能有所不同。建议参考官方文档获取最新信息。