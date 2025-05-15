from ultralytics import YOLO
import cv2
import torch
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torchvision.models import resnet18

# ================== 设备配置 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 模型加载 ==================
# YOLO模型配置
yolo_model_path = 'C:\\Users\\14984\\Desktop\\exp\\runs\\detect\\defect_v8s\\weights\\best.pt'
model_yolo = YOLO(yolo_model_path)

# CNN模型配置
class_names = ['dong', 'que', 'normal']
model_cnn = resnet18(pretrained=False)
model_cnn.fc = torch.nn.Linear(model_cnn.fc.in_features, len(class_names))
model_cnn.load_state_dict(torch.load('defect_cnn.pth', map_location=device))
model_cnn.eval()

# ================== 可视化参数 ==================
COLORS = {
    'logo': (255, 0, 0),    # 蓝色 - 粗框
    'mao': (255, 165, 0),   # 橙色
    'dong': (0, 0, 255),     # 红色 
    'que': (0, 255, 0)       # 绿色
}

STYLE_CONFIG = {
    'logo': {
        'thickness': 4,  # 框线粗细
        'font_scale': 2.4,  # 字体大小（原1.2×2）
        'font_thickness': 5,  # 字体粗细
        'bg_padding': 15  # 文字背景填充
    },
    'default': {
        'thickness': 2,
        'font_scale': 0.8,
        'font_thickness': 2,
        'bg_padding': 5
    }
}

# ================== 核心检测函数 ==================
def detect_all_defects(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None

    # YOLO检测
    yolo_results = model_yolo.predict(img, conf=0.4, verbose=False)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)

    all_defects = []
    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        label = model_yolo.names[cls_id]

        # 记录YOLO检测结果
        all_defects.append({"type": label, "bbox": [x1, y1, x2, y2]})

        # logo区域进行CNN分类
        if label == 'logo':
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                input_tensor = transforms.Resize((224, 224))(crop_pil)
                input_tensor = transforms.ToTensor()(input_tensor)
                input_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_tensor)
                input_tensor = input_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model_cnn(input_tensor)
                    pred = torch.argmax(output).item()

                cnn_class = class_names[pred]
                if cnn_class != 'normal':
                    all_defects.append({"type": cnn_class, "bbox": [x1, y1, x2, y2]})
            except Exception as e:
                print(f"分类异常: {e}")

    return img, all_defects


# ================== 增强绘制函数 ==================
def draw_enhanced_boxes(img, defects):
    for defect in defects:
        x1, y1, x2, y2 = defect['bbox']
        label = defect['type']

        # 获取样式配置
        style = STYLE_CONFIG.get(label, STYLE_CONFIG['default'])

        # 绘制增强框
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[label],
                      style['thickness'])

        # 计算文字尺寸
        (text_width,
         text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           style['font_scale'],
                                           style['font_thickness'])

        # 文字位置计算（确保不超出图像边界）
        text_y = max(y1 - 10, text_height + style['bg_padding'])
        text_x = max(x1, 0)

        # 文字背景框（扩展背景区域）
        cv2.rectangle(img, (text_x - style['bg_padding'] // 2,
                            text_y - text_height - style['bg_padding']),
                      (text_x + text_width + style['bg_padding'] // 2,
                       text_y + style['bg_padding'] // 2), COLORS[label], -1)

        # 绘制超大文字
        cv2.putText(
            img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            style['font_scale'],
            (255, 255, 255),  # 白色文字
            style['font_thickness'],
            lineType=cv2.LINE_AA)
    return img
# ================== 批量处理流程 ==================
def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith(('.bmp', '.jpg', '.png')):
            continue

        img_path = os.path.join(input_dir, img_name)
        print(f"正在处理: {img_path}")

        img, defects = detect_all_defects(img_path)
        if img is None:
            continue

        # 使用增强绘制函数
        img = draw_enhanced_boxes(img, defects)

        output_path = os.path.join(output_dir, f"enhanced_{img_name}")
        cv2.imwrite(output_path, img)
        print(f"结果保存至: {output_path}")

# ================== 执行主程序 ==================
if __name__ == "__main__":
    input_folder = "E:\\2025homework\\exp\\yolo_dataset\\images\\all"
    output_folder = "E:\\2025homework\\exp\\enhanced_results"

    process_folder(input_folder, output_folder)
    print(f"处理完成！结果保存在: {output_folder}")