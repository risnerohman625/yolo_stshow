import os
import json
from PIL import Image

# 分类标签顺序，只保留logo和mao
label_list = ['logo', 'mao']
label2id = {name: i for i, name in enumerate(label_list)}

# 你的数据根路径
json_root = "./data_label"
img_root = "./data_pic"

# 输出路径
out_img_dir = "./yolo_dataset/images/all"
out_label_dir = "./yolo_dataset/labels/all"
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)

# 遍历所有子文件夹
for subfolder in os.listdir(json_root):
    json_subdir = os.path.join(json_root, subfolder)
    if not os.path.isdir(json_subdir):
        continue

    for json_file in os.listdir(json_subdir):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(json_subdir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 对应图像路径（假设bmp与json同名）
        img_name = data['imagePath']
        base_name = os.path.splitext(json_file)[0]
        image_path = None

        # 在所有 data_pic 子目录中查找图像
        for img_subfolder in os.listdir(img_root):
            possible_img = os.path.join(img_root, img_subfolder,
                                        base_name + ".bmp")
            if os.path.exists(possible_img):
                image_path = possible_img
                break

        if not image_path:
            print(f"找不到对应图片：{base_name}.bmp")
            continue

        img = Image.open(image_path)
        img_w, img_h = img.size

        # 输出 YOLO 标签，只处理logo和mao类
        yolo_lines = []
        for shape in data['shapes']:
            label = shape['label']
            if label not in label2id:
                continue
            cls_id = label2id[label]
            points = shape['points']
            x1 = min(p[0] for p in points)
            y1 = min(p[1] for p in points)
            x2 = max(p[0] for p in points)
            y2 = max(p[1] for p in points)
            xc = (x1 + x2) / 2 / img_w
            yc = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:  # 确保有有效的标注才保存
            # 保存图像到新路径
            img.save(os.path.join(out_img_dir, base_name + ".bmp"))

            # 保存 label
            with open(os.path.join(out_label_dir, base_name + ".txt"),
                      "w") as f:
                f.write("\n".join(yolo_lines))