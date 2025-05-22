import streamlit as st
import streamlit.components.v1 as components  # 用来注入 adapter.js
import av
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)

# ====== 1. 注入 adapter.js polyfill （**一定要在任何 WebRTC 调用之前**） ======
#    这样浏览器才会知道 RTCPeerConnection、getUserMedia 等 API
components.html(
    """
    <script src="https://webrtc.github.io/adapter/adapter-latest.js"></script>
    """,
    height=0,
)

# ====== 2. 页面配置（set_page_config 必须最先调用 streamlit 的 API） ======
st.set_page_config(
    page_title="智能质检系统",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 智能质检（WebRTC 版）")

# ====== 3. 模型加载（缓存资源） ======
@st.cache_resource
def load_models():
    # YOLO 检测模型
    yol = YOLO("runs/detect/defect_v8s/weights/best.pt")
    # 一个简单的 CNN 二次分类器
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

# ====== 4. 视频处理器 ======
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

# ====== 5. 侧栏参数 ======
conf = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)

# ====== 6. 启动 WebRTC 流 ======
webrtc_ctx = webrtc_streamer(
    key="yolo-webrtc",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoTransformer,
)

# 实时更新置信度
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.conf_th = conf

# 小提示
st.sidebar.markdown(
    """
    **使用说明**  
    1. 浏览器会弹窗请求摄像头权限，点“允许”启动检测  
    2. 侧栏可调整置信度阈值  
    """
)
