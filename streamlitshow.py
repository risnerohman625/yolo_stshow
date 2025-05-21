import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
from streamlit.components.v1 import html


# ================== 保存结果 ==================
def save_result(defects, frame, manual=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{'manual' if manual else 'auto'}_capture_{timestamp}.jpg"
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", filename)
    cv2.imwrite(save_path, frame)

    if defects:
        main_defect = max(defects, key=lambda x: x["confidence"])
        record = {
            "time": timestamp,
            "file": filename,
            "defect_type": main_defect["type"],
            "confidence": main_defect["confidence"],
        }
    else:
        record = {
            "time": timestamp,
            "file": filename,
            "defect_type": "normal",
            "confidence": 1.0,
        }
    st.session_state.detection_history.append(record)
    st.toast(f"已保存：{filename}")


# ================== 模型加载 ==================
@st.cache_resource
def load_models():
    # YOLO 检测模型
    yolo_model = YOLO("./runs/detect/defect_v8s/weights/best.pt")
    # CNN 细分类模型
    cnn_model = resnet18(pretrained=False)
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, 3)
    cnn_model.load_state_dict(torch.load("defect_cnn.pth", map_location="cpu"))
    cnn_model.eval()
    return yolo_model, cnn_model


yolo_model, cnn_model = load_models()


# ================== 核心检测 ==================
CLASS_NAMES = ["dong", "que", "normal"]


def cnn_classify(crop_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img_t = transform(Image.fromarray(crop_img)).unsqueeze(0)
    with torch.no_grad():
        out = cnn_model(img_t)
    return CLASS_NAMES[int(out.argmax())]


def yolo_detect(frame, conf_threshold):
    res = yolo_model.predict(frame, conf=conf_threshold, verbose=False)[0]
    defects = []
    for box, cls_idx, conf in zip(
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.cls.cpu().numpy().astype(int),
        res.boxes.conf.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)
        label = res.names[cls_idx]
        defects.append({
            "type": label,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf)
        })
        # 对 logo 再跑一次 CNN 细分类
        if label == "logo":
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                cnn_rst = cnn_classify(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if cnn_rst != "normal":
                    defects.append({
                        "type": cnn_rst,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": 0.8
                    })
    return defects


# ================== 可视化 ==================
STYLE_CONFIG = {
    "logo":    {"thickness": 4, "font_scale": 2.4, "font_thickness": 5},
    "default": {"thickness": 2, "font_scale": 0.8, "font_thickness": 2},
}
COLORS = {
    "logo": (255, 0, 0),
    "mao":  (255, 165, 0),
    "dong": (0, 0, 255),
    "que":  (0, 255, 0),
}


def draw_results(frame, defects):
    for d in defects:
        x1, y1, x2, y2 = d["bbox"]
        style = STYLE_CONFIG.get(d["type"], STYLE_CONFIG["default"])
        color = COLORS.get(d["type"], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, style["thickness"])
        (tw, th), _ = cv2.getTextSize(d["type"],
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      style["font_scale"],
                                      style["font_thickness"])
        yy = max(th + 5, y1)
        xx = min(frame.shape[1] - tw, x1)
        cv2.rectangle(frame,
                      (xx, yy - th - 5),
                      (xx + tw, yy),
                      color,
                      -1)
        cv2.putText(frame,
                    d["type"],
                    (xx, yy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    style["font_scale"],
                    (255, 255, 255),
                    style["font_thickness"])
    return frame


# 空格键监听
def add_space_key_listener():
    js = """
    <script>
    document.addEventListener('keydown', e => {
      if (e.code==='Space') {
        window.parent.postMessage(
          {type:'streamlit:setComponentValue', value:'space_pressed'}, '*'
        )
      }
    });
    </script>
    """
    html(js, height=0)


# ================== Streamlit UI ==================
st.set_page_config("智能质检系统", "🔍", layout="wide")
add_space_key_listener()

# 初始化 session_state
st.session_state.setdefault("detection_history", [])
st.session_state.setdefault("video_playing", False)
st.session_state.setdefault("space_pressed", None)

with st.sidebar:
    st.header("系统配置")
    mode = st.radio("检测模式", ["实时摄像头", "上传图片", "上传视频"])
    conf_th = st.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)
    auto_save = st.checkbox("自动保存", False)
    manual_save = st.checkbox("空格手动保存", True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("原画面")
    orig_disp = st.empty()
with col2:
    st.subheader("检测后")
    det_disp = st.empty()

# ———— 主流程略，与原有逻辑相同，只是调用 yolo_detect / draw_results / save_result ————

# 最后展示检测历史
with st.expander("检测历史"):
    if st.session_state.detection_history:
        df = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(df)
    else:
        st.info("暂无历史记录")
