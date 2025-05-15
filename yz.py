from ultralytics import YOLO
import os

model = YOLO('E:\\2025homework\\exp\\runs\\detect\\defect_v8s\\weights\\best.pt')
results = model("E:\\2025homework\\exp\\data_pic\\problem\\Image173.bmp",
                save=True)  # 保存带检测框的图片
