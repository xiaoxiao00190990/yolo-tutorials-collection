# YOLO 训练与部署教程大全

本仓库收集了YOLO（You Only Look Once）目标检测模型的训练和部署相关教程、代码示例和最佳实践。

## 📚 目录
- [训练教程](#训练教程)
- [部署教程](#部署教程)
- [中文资源](#中文资源)
- [项目模板](#项目模板)
- [快速开始](#快速开始)

## 训练教程

### 1. 基础训练教程
- **[jkjung-avt/yolov4_crowdhuman](https://github.com/jkjung-avt/yolov4_crowdhuman)** - 使用DarkNet YOLOv4训练CrowdHuman数据集的完整教程
- **[berktepebag/Traffic-light-detection-with-YOLOv3-BOSCH-traffic-light-dataset](https://github.com/berktepebag/Traffic-light-detection-with-YOLOv3-BOSCH-traffic-light-dataset)** - 使用BOSCH交通灯数据集训练YOLOv3
- **[KleinYuan/easy-yolo](https://github.com/KleinYuan/easy-yolo)** - 使用深度学习神经网络进行YOLO模型训练的教程

### 2. 高级训练教程
- **[WyattAutomation/Train-YOLOv3-with-OpenImagesV4](https://github.com/WyattAutomation/Train-YOLOv3-with-OpenImagesV4)** - 使用Google Open Images V4数据集训练YOLOv3的端到端教程
- **[cj-mills/pytorch-yolox-object-detection-tutorial-code](https://github.com/cj-mills/pytorch-yolox-object-detection-tutorial-code)** - PyTorch YOLOX对象检测训练代码
- **[roboflow/yolov5-custom-training-tutorial](https://github.com/roboflow/yolov5-custom-training-tutorial)** - YOLOv5自定义训练教程

### 3. 特定应用训练
- **[umairalam289/Yolov8_Tutorial_Fish_Detection](https://github.com/umairalam289/Yolov8_Tutorial_Fish_Detection)** - 使用YOLOv8进行鱼类检测
- **[Poyqraz/YOLO-v8-Object-Detection-Tutorial-on-CPU-GPU](https://github.com/Poyqraz/YOLO-v8-Object-Detection-Tutorial-on-CPU-GPU)** - YOLOv8在CPU和GPU上的对象检测训练
- **[Nannigalaxy/facemask_detection_yolov5](https://github.com/Nannigalaxy/facemask_detection_yolov5)** - 使用YOLOv5进行口罩检测

## 部署教程

### 1. 通用部署
- **[EdjeElectronics/Train-and-Deploy-YOLO-Models](https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models)** - Ultralytics YOLO模型训练和部署教程（⭐ 254 stars）
- **[FranciscoReveriano/YOLOV3-Tutorial](https://github.com/FranciscoReveriano/YOLOV3-Tutorial)** - 简单YOLOv3模型，可直接部署

### 2. 边缘设备部署
- **[mrtj/yolox-panorama-tutorial](https://github.com/mrtj/yolox-panorama-tutorial)** - 在AWS Panorama设备上部署自定义YOLOX模型
- **[1760hwy/YOLO11n-INT8](https://github.com/1760hwy/YOLO11n-INT8)** - RDKX5 YOLO11n-INT8全流程量化部署教程
- **[WSJ261126/RDK-X5-YOLOv5-Deploy](https://github.com/WSJ261126/RDK-X5-YOLOv5-Deploy)** - 地平线RDK X5部署YOLOv5全流程指南

### 3. 特定平台部署
- **[Psicodelic/YOLO11-Edge](https://github.com/Psicodelic/YOLO11-Edge)** - 在Horizon X5 RDK边缘设备上部署YOLO11
- **[767172261/Yolov8-rotating-target-detection-deployment-tutorial-with-code-c_python](https://github.com/767172261/Yolov8-rotating-target-detection-deployment-tutorial-with-code-c_python)** - YOLOv8旋转目标检测部署教程（C++/Python）

## 中文资源

### 1. 中文教程
- **[2585157341/YOLO_v3_tutorial_from_scratch-master_Chinese_note](https://github.com/2585157341/YOLO_v3_tutorial_from_scratch-master_Chinese_note)** - YOLOv3从零开始教程中文注释版

### 2. 中文部署指南
- **[WSJ261126/RDK-X5-YOLOv5-Deploy](https://github.com/WSJ261126/RDK-X5-YOLOv5-Deploy)** - 地平线RDK X5部署YOLOv5全流程指南（中英双语）
- **[767172261/yolov8-robot-key-point-detection-model-deployment-tutorial-code-dataset-industrial-application](https://github.com/767172261/yolov8-robot-key-point-detection-model-deployment-tutorial-code-dataset-industrial-application)** - YOLOv8机械臂关键点检测模型部署+教程+代码+数据集

## 项目模板

### 快速开始模板
```bash
# 克隆模板项目
git clone https://github.com/your-username/yolo-quickstart-template.git
cd yolo-quickstart-template

# 安装依赖
pip install -r requirements.txt

# 准备数据
python prepare_data.py --dataset your_dataset

# 训练模型
python train.py --model yolov8n --epochs 100 --data data.yaml

# 部署模型
python deploy.py --weights runs/train/exp/weights/best.pt
```

### 模板包含内容
- 数据准备脚本
- 训练配置
- 模型评估
- 部署脚本
- Docker支持
- 本地网络部署指南

## 快速开始

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv yolo-env
source yolo-env/bin/activate

# 安装基础依赖
pip install torch torchvision
pip install ultralytics  # YOLOv8
pip install opencv-python
pip install matplotlib
```

### 2. 数据准备
创建`data.yaml`文件：
```yaml
# 数据集配置
path: ./datasets/custom
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 2

# 类别名称
names: ['class1', 'class2']
```

### 3. 训练模型
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'  # 或 'cpu'
)
```

### 4. 模型部署
```python
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/train/exp/weights/best.pt')

# 推理
results = model('image.jpg')

# 显示结果
results[0].show()
```

## 📖 学习路径建议

### 初学者
1. 从YOLOv8开始（最简单）
2. 学习基础训练流程
3. 尝试自定义数据集训练
4. 学习模型评估和优化

### 中级开发者
1. 学习YOLOv5/v8的部署
2. 掌握模型量化技术
3. 学习边缘设备部署
4. 了解模型优化技巧

### 高级开发者
1. 研究YOLO架构原理
2. 学习模型剪枝和蒸馏
3. 掌握多平台部署
4. 贡献开源项目

## 🔗 相关资源

### 官方文档
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [OpenCV Documentation](https://docs.opencv.org/)

### 社区资源
- [Roboflow Blog](https://blog.roboflow.com/)
- [Towards Data Science](https://towardsdatascience.com/)
- [PyImageSearch](https://pyimagesearch.com/)

### 中文社区
- [CSDN YOLO专栏](https://blog.csdn.net/nav/ai)
- [知乎计算机视觉话题](https://www.zhihu.com/topic/19559424)
- [Bilibili AI教程](https://www.bilibili.com/video/BV1fX4y1g7wx)

## 🤝 贡献指南

欢迎提交PR来完善这个资源列表！

1. Fork本仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

---

**最后更新**: 2025年4月23日  
**维护者**: [550923629](https://github.com/550923629)  
**仓库地址**: https://github.com/550923629/yolo-tutorials-collection