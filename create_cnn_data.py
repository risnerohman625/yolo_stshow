from ultralytics import YOLO
import cv2
import os

# 加载YOLO模型
model_yolo = YOLO(
    'E:\\2025homework\\exp\\runs\\detect\\defect_v8s\\weights\\best.pt')

# 遍历原始图片
src_img_dir = "./yolo_dataset/images/all"
dst_cnn_dir = "./cnn_dataset"
os.makedirs(dst_cnn_dir, exist_ok=True)

for img_name in os.listdir(src_img_dir):
    img_path = os.path.join(src_img_dir, img_name)
    img = cv2.imread(img_path)

    # YOLO检测logo
    results = model_yolo.predict(img, conf=0.7)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取检测框

    # 裁剪每个logo区域
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(
            f"{dst_cnn_dir}/{os.path.splitext(img_name)[0]}_crop{i}.bmp", crop)
