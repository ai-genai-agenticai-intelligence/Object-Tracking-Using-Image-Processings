import streamlit as st
import cv2
import tempfile

st.set_page_config(
    page_title="Object Tracking App",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Object Tracking using Image Processing")
st.write("Upload a video to detect moving objects using Background Subtraction (MOG2).")

# Upload video
uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    object_detector = cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=40
    )

    col1, col2 = st.columns(2)

    frame_placeholder = col1.empty()
    mask_placeholder = col2.empty()

    st.success("Processing video...")

    stop = st.button("⏹ Stop")

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret or stop:
            break

        # Background subtraction
        mask = object_detector.apply(frame)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            caption="Original Frame",
            use_container_width=True,
        )

        mask_placeholder.image(
            mask,
            caption="Detection Mask",
            use_container_width=True,
        )

    cap.release()
    st.success("✅ Video processing completed!")

else:
    st.info("Please upload a video file.")
