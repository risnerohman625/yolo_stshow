import os
import shutil
import random

image_dir = "./yolo_dataset/images/all"
label_dir = "./yolo_dataset/labels/all"

train_img_dir = "./yolo_dataset/images/train"
val_img_dir = "./yolo_dataset/images/val"
train_label_dir = "./yolo_dataset/labels/train"
val_label_dir = "./yolo_dataset/labels/val"

# 创建目录
for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    os.makedirs(d, exist_ok=True)

all_images = [f for f in os.listdir(image_dir) if f.endswith(".bmp")]
random.shuffle(all_images)

split = int(0.8 * len(all_images))
train_files = all_images[:split]
val_files = all_images[split:]

for f in train_files:
    shutil.copy(os.path.join(image_dir, f), os.path.join(train_img_dir, f))
    shutil.copy(os.path.join(label_dir, f.replace(".bmp", ".txt")),
                os.path.join(train_label_dir, f.replace(".bmp", ".txt")))

for f in val_files:
    shutil.copy(os.path.join(image_dir, f), os.path.join(val_img_dir, f))
    shutil.copy(os.path.join(label_dir, f.replace(".bmp", ".txt")),
                os.path.join(val_label_dir, f.replace(".bmp", ".txt")))
