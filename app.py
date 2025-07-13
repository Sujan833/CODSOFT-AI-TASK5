import cv2
import streamlit as st
import numpy as np
from PIL import Image

CASCADE_PATH = "haarcascade_frontalface_default.xml"

st.title("Real-time Face Detection App")
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if ext in ["jpg", "jpeg", "png"]:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Faces", use_column_width=True)

    elif ext == "mp4":
        st.video(uploaded_file)
        st.warning("Real-time video face detection is limited on Streamlit Cloud.\nYou can only show the video, not process it in real time.")
