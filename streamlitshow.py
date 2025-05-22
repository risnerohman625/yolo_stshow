import os
import time
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
from streamlit.components.v1 import html
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# ================== 全局配置 ==================
CLASS_NAMES = ['dong', 'que', 'normal']
STYLE_CONFIG = {
    'logo':   {'thickness': 4, 'font_scale': 2.4, 'font_thickness': 5},
    'default':{'thickness': 2, 'font_scale': 0.8, 'font_thickness': 2},
}
COLORS = {
    'logo': (255,0,0), 'mao': (255,165,0),
    'dong': (0,0,255), 'que': (0,255,0),
}

# ================== 动态路径 ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
YOLO_PATH = os.path.join(MODEL_DIR, "best.pt")
CNN_PATH  = os.path.join(MODEL_DIR, "defect_cnn.pth")

# ================== 加载模型 ==================
@st.cache_resource
def load_models():
    if not os.path.exists(YOLO_PATH):
        st.error(f"未找到 YOLO 模型：{YOLO_PATH}")
    if not os.path.exists(CNN_PATH):
        st.error(f"未找到 CNN 模型：{CNN_PATH}")
    yolo = YOLO(YOLO_PATH)
    cnn = resnet18(pretrained=False)
    cnn.fc = torch.nn.Linear(cnn.fc.in_features, len(CLASS_NAMES))
    cnn.load_state_dict(torch.load(CNN_PATH, map_location="cpu"))
    cnn.eval()
    return yolo, cnn

yolo_model, cnn_model = load_models()

# ================== 保存结果 ==================
def save_result(defects, frame, manual=False):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"{'manual' if manual else 'auto'}_capture_{ts}.jpg"
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", fn)
    cv2.imwrite(path, frame)
    if defects:
        m = max(defects, key=lambda d: d['confidence'])
        rec = {"time":ts,"file":fn,
               "defect_type":m['type'],"confidence":m['confidence']}
    else:
        rec = {"time":ts,"file":fn,"defect_type":"正常","confidence":1.0}
    st.session_state.detection_history.append(rec)
    st.toast(f"已保存检测结果：{fn}")

# ================== 检测逻辑 ==================
def cnn_classify(crop):
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    x = tf(Image.fromarray(crop)).unsqueeze(0)
    with torch.no_grad():
        out = cnn_model(x)
    return CLASS_NAMES[int(out.argmax())]

def yolo_detect(frame, conf):
    res = yolo_model.predict(frame, conf=conf, verbose=False)[0]
    defects = []
    for box, cls, cf in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.cls.cpu().numpy().astype(int),
                             res.boxes.conf.cpu().numpy()):
        x1,y1,x2,y2 = map(int, box)
        lbl = res.names[cls]
        defects.append({"type":lbl,"bbox":[x1,y1,x2,y2],"confidence":float(cf)})
        if lbl=="logo":
            crop = frame[y1:y2,x1:x2]
            if crop.size>0:
                sub = cnn_classify(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
                if sub!="normal":
                    defects.append({"type":sub,"bbox":[x1,y1,x2,y2],"confidence":0.8})
    return defects

def draw_results(frame, defects):
    for d in defects:
        x1,y1,x2,y2 = d['bbox']; lab=d['type']
        s = STYLE_CONFIG.get(lab,STYLE_CONFIG['default'])
        cv2.rectangle(frame,(x1,y1),(x2,y2), COLORS[lab], s['thickness'])
        (tw,th),_ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX,
                                   s['font_scale'], s['font_thickness'])
        yy = max(th+5, y1)
        cv2.rectangle(frame,(x1,yy-th-5),(x1+tw,yy), COLORS[lab], -1)
        cv2.putText(frame, lab, (x1,yy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, s['font_scale'],
                    (255,255,255), s['font_thickness'])
    return frame

# ================== webrtc 处理器 ==================
class YoloProcessor(VideoProcessorBase):
    def __init__(self, conf): self.conf=conf
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        det = yolo_detect(img, self.conf)
        out = draw_results(img.copy(), det)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

# ================== 空格监听 ==================
def add_space_listener():
    js = """
    <script>
      document.addEventListener('keydown', e=>{
        if(e.code==='Space'){
          window.parent.postMessage(
            {type:'streamlit:setComponentValue',value:'space_pressed'},'*');
        }
      });
    </script>
    """
    html(js, height=0, width=0)

# ================== 主界面 ==================
st.set_page_config("智能质检系统","🔍",layout="wide")
add_space_listener()
os.makedirs("data", exist_ok=True)
st.session_state.setdefault("detection_history", [])

with st.sidebar:
    st.header("系统配置")
    mode = st.radio("检测模式", ["实时摄像头","上传图片","上传视频"])
    conf = st.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)
    auto = st.checkbox("自动保存检测结果", False)
    manual = st.checkbox("空格键手动保存", True)

st.title("🔍 智能质检系统")
c1,c2 = st.columns(2)
orig = c1.empty(); det = c2.empty()

if mode=="实时摄像头":
    webrtc_streamer(
        key="yolo",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: YoloProcessor(conf),
        async_processing=True,
    )
    st.info("请授权浏览器访问摄像头")

elif mode=="上传图片":
    up = st.file_uploader("上传检测图片", type=["jpg","png","jpeg","bmp"])
    if up:
        img = np.array(Image.open(up).convert("RGB"))
        dets = yolo_detect(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),conf)
        out = draw_results(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),dets)
        orig.image(img,use_column_width=True,caption="原图")
        det.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB),
                  use_column_width=True,caption="检测结果")
        if auto:    save_result(dets,out)
        if manual and st.button("保存当前图片"):
            save_result(dets,out,manual=True)

elif mode=="上传视频":
    vid = st.file_uploader("上传检测视频", type=["mp4","avi"])
    if vid:
        tmp="temp.mp4"
        with open(tmp,"wb") as f: f.write(vid.read())
        cap=cv2.VideoCapture(tmp)
        st.session_state.setdefault("playing",False)
        if st.button("播放/暂停"):
            st.session_state.playing = not st.session_state.playing
        while st.session_state.playing and cap.isOpened():
            ret,frm = cap.read()
            if not ret: break
            dets=yolo_detect(frm,conf)
            out = draw_results(frm.copy(),dets)
            orig.image(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB),
                       use_column_width=True)
            det.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB),
                      use_column_width=True)
            if auto: save_result(dets,out)
            if manual and st.button("保存帧"):
                save_result(dets,out,manual=True)
        cap.release(); os.remove(tmp)

with st.expander("检测结果统计"):
    if st.session_state.detection_history:
        last = st.session_state.detection_history[-1]
        st.metric("最新缺陷类型", last['defect_type'])
        st.metric("置信度", f"{last['confidence']*100:.1f}%")
        st.dataframe(pd.DataFrame(st.session_state.detection_history),
                     use_container_width=True)
    else:
        st.info("暂无检测数据")
