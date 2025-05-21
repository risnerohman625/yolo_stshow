import av
import cv2
import torch
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
    ClientSettings,
)
from torchvision import transforms
from ultralytics import YOLO

# ————— 页面配置 —————
st.set_page_config(
    page_title="智能质检系统",
    page_icon="🔍",
    layout="wide",
)
st.title("🔍 智能质检（WebRTC 版）")

# ————— 模型加载 —————
@st.cache_resource
def load_models():
    # YOLO 检测模型（请换成你自己的权重路径）
    yolo = YOLO("runs/detect/defect_v8s/weights/best.pt")
    # CNN 分类模型示例：ResNet18
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

# ————— 视频处理器 —————
class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.conf_th = 0.5

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = yolo_model.predict(img, conf=self.conf_th, verbose=False)[0]
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            label = results.names[int(cls.cpu().numpy())]
            if label == "defect":
                crop = img[y1:y2, x1:x2]
                sublabel = cnn_classify(crop)
                label = f"{label}/{sublabel}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                img, f"{label} {conf:.2f}",
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 2
            )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ————— 侧栏 —————
conf = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)

# ————— 启动 WebRTC —————
webrtc_ctx = webrtc_streamer(
    key="yolo-webrtc",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
    ),
    video_processor_factory=VideoTransformer,
)

# 实时更新置信度
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.conf_th = conf

st.sidebar.markdown(
    """
    **使用说明**  
    1. 运行后浏览器会请求“允许访问摄像头”。  
    2. 点击“允许”即可开始实时检测。  
    3. 侧栏可调整 Detection Confidence。  
    """
)
