import streamlit as st
import cv2
import tempfile

st.set_page_config(
    page_title="Object Tracking",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Object Tracking using Image Processing")
st.write("Upload a video and detect moving objects using OpenCV Background Subtraction (MOG2).")

# Upload video
uploaded_file = st.file_uploader(
    "Upload Highway Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        st.stop()

    # Sidebar settings
    st.sidebar.header("⚙ Detection Settings")

    history = st.sidebar.slider("History", 50, 500, 100)
    threshold = st.sidebar.slider("Var Threshold", 10, 100, 40)
    speed = st.sidebar.slider("Playback Delay (ms)", 1, 100, 30)

    detector = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=threshold
    )

    col1, col2 = st.columns(2)

    frame_placeholder = col1.empty()
    mask_placeholder = col2.empty()

    start = st.button("▶ Start Tracking")

    if start:
        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                st.success("Video completed!")
                break

            mask = detector.apply(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(frame_rgb, caption="Original Video", use_container_width=True)

            mask_placeholder.image(mask, caption="Foreground Mask", use_container_width=True, clamp=True)

            cv2.waitKey(speed)

    cap.release()

else:
    st.info("👆 Upload a highway video to begin object tracking.")