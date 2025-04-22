import streamlit as st
from PIL import Image
import tempfile
import os
from pipeline import VisionLanguageDrivingPipeline  # Make sure this is the final integrated pipeline
import cv2

# Initialize pipeline
pipeline = VisionLanguageDrivingPipeline()

st.set_page_config(page_title="🚘 Vision-Language Autonomous Driving Assistant", layout="centered")

st.title(":oncoming_automobile: Vision-Language Autonomous Driving Assistant")
st.markdown("Choose Input Mode")

mode = st.radio("", ["Upload Image", "Capture with Webcam", "Dashcam Mode (Video)"])

if mode == "Upload Image":
    st.markdown("### :arrow_up: Upload a driving scene image")
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Scene", use_column_width=True)

        with st.spinner("Analyzing scene with vision-language pipeline..."):
            result = pipeline.process_image(image)

        st.markdown("---")
        st.subheader("**Scene Analysis Output**")
        st.write("**Caption (BLIP2):**", result["caption"])
        st.write("**Objects Detected (DETR):**", result["objects"])
        st.write("**Scene Reasoning (GPT-2):**", result["reasoning"])
        st.write("**Driving Decision (VectorBC):**", result["decision"])

elif mode == "Capture with Webcam":
    st.markdown("### 🚀 Capture image from webcam")
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")
        st.image(image, caption="Captured Scene", use_column_width=True)

        with st.spinner("Analyzing scene with vision-language pipeline..."):
            result = pipeline.process_image(image)

        st.markdown("---")
        st.subheader("**Scene Analysis Output**")
        st.write("**Caption (BLIP2):**", result["caption"])
        st.write("**Objects Detected (DETR):**", result["objects"])
        st.write("**Scene Reasoning (GPT-2):**", result["reasoning"])
        st.write("**Driving Decision (VectorBC):**", result["decision"])

elif mode == "Dashcam Mode (Video)":
    st.markdown("### 🎥 Upload dashcam video (10s clips)")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        st.video(video_path)
        st.write(f"Video Duration: {duration:.2f} seconds")

        frame_number = st.slider("Choose second to analyze (1 frame per sec)", 0, int(duration) - 1, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number * fps)
        ret, frame = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            st.image(image, caption=f"Frame at {frame_number} sec", use_column_width=True)

            with st.spinner("Analyzing scene with vision-language pipeline..."):
                result = pipeline.process_image(image)

            st.markdown("---")
            st.subheader("**Scene Analysis Output**")
            st.write("**Caption (BLIP2):**", result["caption"])
            st.write("**Objects Detected (DETR):**", result["objects"])
            st.write("**Scene Reasoning (GPT-2):**", result["reasoning"])
            st.write("**Driving Decision (VectorBC):**", result["decision"])

        cap.release()

# Footer
st.markdown("---")
st.markdown("Final Year Project by Tushar And Amrit Raj Paramhans")
