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
)
from torchvision import transforms
from ultralytics import YOLO

# ========= 页面配置 =========
st.set_page_config(
    page_title="智能质检系统",
    page_icon="🔍",
    layout="wide",
)
st.title("🔍 智能质检（WebRTC 版）")

# ========= 模型加载 =========
@st.cache_resource
def load_models():
    # 1) YOLO 检测
    yol = YOLO("runs/detect/defect_v8s/weights/best.pt")
    # 2) CNN 分类示例（ResNet18）
    cnn = torch.hub.load("pytorch/vision:v0.14.0", "resnet18", pretrained=False)
    cnn.fc = torch.nn.Linear(cnn.fc.in_features, 3)
    cnn.load_state_dict(torch.load("defect_cnn.pth", map_location="cpu"))
    cnn.eval()
    return yol, cnn

yolo_model, cnn_model = load_models()
CLASS_NAMES = ["dong", "que", "normal"]

def cnn_classify(crop: np.ndarray) -> str:
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    tensor = tf(Image.fromarray(crop)).unsqueeze(0)
    with torch.no_grad():
        out = cnn_model(tensor)
    return CLASS_NAMES[int(out.argmax())]

# ========= 视频处理器 =========
class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.conf_th = 0.5

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = yolo_model.predict(img, conf=self.conf_th, verbose=False)[0]
        for box, cls, conf in zip(
            results.boxes.xyxy, results.boxes.cls, results.boxes.conf
        ):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            label = results.names[int(cls.cpu().numpy())]
            if label == "defect":
                crop = img[y1:y2, x1:x2]
                sub = cnn_classify(crop)
                label = f"{label}/{sub}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2,
            )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========= 侧栏参数 =========
conf = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)

# ========= 启动 WebRTC =========
webrtc_streamer(
    key="yolo-webrtc",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoTransformer,
)

# 动态更新阈值
if st.session_state.get("webrtc_context") and \
   st.session_state.webrtc_context.video_processor:
    st.session_state.webrtc_context.video_processor.conf_th = conf

st.sidebar.markdown(
    """
    **使用说明**  
    1. 浏览器会弹窗请求摄像头权限，点“允许”启动检测  
    2. 侧栏可调整置信度阈值  
    """
)
