import av
import cv2
import torch
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from torchvision import transforms
from ultralytics import YOLO

# ———————— 页面配置 ————————
st.set_page_config(
    page_title="智能质检系统", 
    page_icon="🔍", 
    layout="wide"
)

st.title("🔍 智能质检（WebRTC 版）")

# ———————— 模型加载 ————————
@st.cache_resource
def load_models():
    # YOLO 检测模型
    yolo = YOLO("runs/detect/defect_v8s/weights/best.pt")
    # CNN 分类模型（示例用 ResNet18）
    cnn = torch.hub.load("pytorch/vision:v0.14.0", "resnet18", pretrained=False)
    cnn.fc = torch.nn.Linear(cnn.fc.in_features, 3)
    cnn.load_state_dict(torch.load("defect_cnn.pth", map_location="cpu"))
    cnn.eval()
    return yolo, cnn

yolo_model, cnn_model = load_models()
CLASS_NAMES = ["dong", "que", "normal"]

def cnn_classify(crop: np.ndarray) -> str:
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tensor = tf(Image.fromarray(crop)).unsqueeze(0)
    with torch.no_grad():
        out = cnn_model(tensor)
    return CLASS_NAMES[int(out.argmax())]

# ———————— 视频帧处理器 ————————
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.conf_th = 0.5

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # YOLO 预测
        result = yolo_model.predict(img, conf=self.conf_th, verbose=False)[0]
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            label = result.names[int(cls.cpu().numpy())]
            # 如果是可疑缺陷类别，再做 CNN 进一步分类
            if label == "defect":
                crop = img[y1:y2, x1:x2]
                sublabel = cnn_classify(crop)
                label = f"{label}/{sublabel}"
            # 绘制框和文字
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(
                img, f"{label} {conf:.2f}", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255,255,255), 2
            )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ———————— 侧栏设置 ————————
conf = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)

# ———————— 启动 WebRTC 流 ————————
webrtc_ctx = webrtc_streamer(
    key="yolo-webrtc",
    mode="SENDRECV",
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_transformer_factory=VideoTransformer,
)

# 动态修改置信度
if webrtc_ctx.video_transformer:
    webrtc_ctx.video_transformer.conf_th = conf

st.sidebar.markdown(
    """
    **使用说明**  
    1. 运行后浏览器会请求“允许访问摄像头”。  
    2. 点击“允许”即可看到画面并开始实时检测。  
    3. 可在侧栏调整 Detection Confidence。  
    """
)
