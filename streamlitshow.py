import os
import time
import threading
from queue import Queue
from datetime import datetime
from io import BytesIO

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import av
from streamlit.components.v1 import html
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

import torch
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
import mss

# ================== 全局配置 ==================
CLASS_NAMES = ['dong', 'que', 'normal']
COLORS = {
    'logo': (255, 0, 0),
    'mao': (255, 165, 0),
    'dong': (0, 0, 255),
    'que': (0, 255, 0)
}
STYLE_CONFIG = {
    'logo': {'thickness': 4, 'font_scale': 2.4, 'font_thickness': 5},
    'default': {'thickness': 2, 'font_scale': 0.8, 'font_thickness': 2}
}

# ================== 路径与模型加载 ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_PATH = os.path.join(MODELS_DIR, "best.pt")
CNN_PATH = os.path.join(MODELS_DIR, "defect_cnn.pth")

@st.cache_resource
def load_models():
    # 动态找文件
    if not os.path.exists(YOLO_PATH) or not os.path.exists(CNN_PATH):
        st.error("请确保 models/best.pt 和 models/defect_cnn.pth 已上传到仓库 models/ 目录下")
    yolo_model = YOLO(YOLO_PATH)
    cnn_model = resnet18(pretrained=False)
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, len(CLASS_NAMES))
    cnn_model.load_state_dict(torch.load(CNN_PATH, map_location='cpu'))
    cnn_model.eval()
    return yolo_model, cnn_model

yolo_model, cnn_model = load_models()

# ================== 检测与绘图 ==================
def cnn_classify(crop_img):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    x = tf(Image.fromarray(crop_img)).unsqueeze(0)
    with torch.no_grad():
        out = cnn_model(x)
    return CLASS_NAMES[int(torch.argmax(out))]

def yolo_detect(frame, conf_thresh):
    results = yolo_model.predict(frame, conf=conf_thresh, verbose=False)[0]
    defects = []
    for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                              results.boxes.cls.cpu().numpy().astype(int),
                              results.boxes.conf.cpu().numpy()):
        x1,y1,x2,y2 = map(int,box)
        label = results.names[cls]
        defects.append({"type":label,"bbox":[x1,y1,x2,y2],"confidence":float(conf)})
        if label=='logo':
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                sub = cnn_classify(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
                if sub!='normal':
                    defects.append({"type":sub,"bbox":[x1,y1,x2,y2],"confidence":0.8})
    return defects

def draw_results(frame, defects):
    for d in defects:
        x1,y1,x2,y2 = d['bbox']
        style = STYLE_CONFIG.get(d['type'], STYLE_CONFIG['default'])
        cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS[d['type']],style['thickness'])
        (tw,th),_ = cv2.getTextSize(d['type'], cv2.FONT_HERSHEY_SIMPLEX,
                                    style['font_scale'], style['font_thickness'])
        cv2.rectangle(frame,(x1,y1-th-5),(x1+tw,y1),COLORS[d['type']],-1)
        cv2.putText(frame,d['type'],(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, style['font_scale'],
                    (255,255,255), style['font_thickness'])
    return frame

# ================== 保存结果 ==================
def save_result(defects, frame, manual=False):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"{'manual' if manual else 'auto'}_{ts}.jpg"
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", fn)
    cv2.imwrite(path, frame)
    rec = {
        "time": ts,
        "file": fn,
        "defect_type": max(defects, key=lambda x:x['confidence'])['type'] if defects else "normal",
        "confidence": max([d['confidence'] for d in defects], default=1.0)
    }
    st.session_state.history.append(rec)

# ================== WebRTC 处理器 ==================
class YoloProcessor(VideoProcessorBase):
    def __init__(self, conf_threshold):
        self.conf = conf_threshold
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        defects = yolo_detect(img, self.conf)
        out = draw_results(img.copy(), defects)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

# ================== 屏幕捕获线程 ==================
frame_q = Queue(1)
det_q = Queue(1)
stop_flag = threading.Event()

def screen_thread(conf, fps, region):
    with mss.mss() as sct:
        while not stop_flag.is_set():
            shot = np.array(sct.grab(region))
            img = cv2.cvtColor(shot,cv2.COLOR_BGRA2BGR)
            det = yolo_detect(img, conf)
            out = draw_results(img.copy(), det)
            if frame_q.full(): frame_q.get_nowait()
            if det_q.full(): det_q.get_nowait()
            frame_q.put(img); det_q.put(out)
            time.sleep(1/fps)

# ================== Streamlit UI ==================
def main():
    st.set_page_config("智能质检系统", "🔍", layout="wide")
    # 初始化 state
    st.session_state.setdefault("history", [])
    st.sidebar.header("设置")
    mode = st.sidebar.radio("模式", ["摄像头(webrtc)", "上传图片", "上传视频", "屏幕捕获"])
    conf = st.sidebar.slider("置信度",0.0,1.0,0.5,0.01)
    auto = st.sidebar.checkbox("自动保存", False)
    manual = st.sidebar.checkbox("空格手动保存", True)

    st.title("🔍 智能质检系统")
    col1,col2 = st.columns(2)
    orig = col1.empty(); det = col2.empty()

    if mode=="摄像头(webrtc)":
        webrtc_streamer(
            key="yolo",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: YoloProcessor(conf),
            async_processing=True
        )
        st.info("请授权浏览器使用摄像头")
    elif mode=="上传图片":
        f = st.file_uploader("图片", type=["jpg","png","jpeg"])
        if f:
            img = np.array(Image.open(f).convert("RGB"))
            d = yolo_detect(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),conf)
            out = draw_results(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),d)
            orig.image(img,use_column_width=True)
            det.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB),use_column_width=True)
            if auto: save_result(d,out)
            if manual and st.button("保存图片"): save_result(d,out,True)
    elif mode=="上传视频":
        f = st.file_uploader("视频", type=["mp4","avi"])
        if f:
            tmp="temp.mp4"
            with open(tmp,"wb") as o: o.write(f.read())
            cap = cv2.VideoCapture(tmp)
            play = st.button("播放")
            if play:
                while cap.isOpened():
                    ret,frm=cap.read()
                    if not ret: break
                    d=yolo_detect(frm,conf)
                    o=draw_results(frm.copy(),d)
                    orig.image(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB),use_column_width=True)
                    det.image(cv2.cvtColor(o,cv2.COLOR_BGR2RGB),use_column_width=True)
                    if auto: save_result(d,o)
                    if manual and st.button("保存帧"): save_result(d,o,True)
                cap.release()
                os.remove(tmp)
    else:  # 屏幕捕获
        fps = st.sidebar.slider("FPS",1,30,10)
        region = {
            "top":st.sidebar.number_input("Top",0,10000,100),
            "left":st.sidebar.number_input("Left",0,10000,100),
            "width":st.sidebar.number_input("W",100,2000,800),
            "height":st.sidebar.number_input("H",100,2000,600),
        }
        if st.button("开始捕获"):
            stop_flag.clear()
            threading.Thread(target=screen_thread, args=(conf,fps,region), daemon=True).start()
        if st.button("停止捕获"):
            stop_flag.set()
        if not frame_q.empty():
            orig.image(cv2.cvtColor(frame_q.get(),cv2.COLOR_BGR2RGB),use_column_width=True)
        if not det_q.empty():
            img=det_q.get()
            det.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),use_column_width=True)
            if auto: save_result(yolo_detect(img,conf),img)
            if manual and st.button("保存帧"): save_result(yolo_detect(img,conf),img,True)

    # 历史记录
    with st.expander("🔖 检测历史"):
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df)
            if st.button("导出Excel"):
                buf=BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                    df.to_excel(w,index=False)
                b64 = base64.b64encode(buf.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="history.xlsx">下载</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("暂无记录")

if __name__=="__main__":
    main()
