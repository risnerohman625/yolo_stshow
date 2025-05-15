from ultralytics import YOLO
import os

# 确认工作目录正确
print("当前工作目录：", os.getcwd())  # 应为 E:/2025homework/yolo

# 加载预训练模型（YOLOv8s平衡型）
model = YOLO('yolo11n.pt')  # 确保模型文件在项目根目录或提供完整路径

# 启动训练
results = model.train(
    data='data.yaml',  # 配置文件路径，确保类别数和名称已更新
    epochs=140,  # 减少epoch防止过拟合（假设数据集约500张）
    imgsz=640,  # 输入分辨率
    batch=10,  # 根据GPU显存调整（RTX 3060建议设为8-16）‘
    patience=15,  # 早停等待轮次
    optimizer='AdamW',  # 优化器选择
    lr0=0.01,  # 初始学习率
    cos_lr=True,  # 启用余弦退火学习率
    device='cpu',  # 使用GPU 0（CPU训练改为device='cpu'）
    workers=4,  # 数据加载线程数
    name='defect_v8s'  # 实验名称（结果保存在runs/detect/defect_v8s）
)