import streamlit as st
import cv2
import tempfile
import os
from tracker import EuclideanDistTracker

st.set_page_config(
    page_title="Object Tracking App",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Object Tracking using Image Processing")
st.write("Detect and track moving objects using **Background Subtraction (MOG2)** and **Euclidean Distance Centroid Tracking**.")

# Video source selection
video_source = st.radio(
    "Select Video Source:",
    options=["Demo Video (highway.mp4)", "Upload Custom Video"],
    horizontal=True
)

video_path = None

if video_source == "Demo Video (highway.mp4)":
    demo_path = "highway.mp4"
    if os.path.exists(demo_path):
        video_path = demo_path
    else:
        st.error(f"Demo video file '{demo_path}' not found in the workspace directory.")
else:
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name

if video_path is not None:
    col_start, col_stop, _ = st.columns([1, 1, 4])
    start_button = col_start.button("▶ Start Tracking", type="primary", use_container_width=True)
    stop_button = col_stop.button("⏹ Stop Tracking", type="secondary", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📹 Video Tracking")
        frame_placeholder = st.empty()
    with col2:
        st.subheader("🎭 Detection Mask (MOG2)")
        mask_placeholder = st.empty()

    if start_button:
        cap = cv2.VideoCapture(video_path)
        tracker = EuclideanDistTracker()
        object_detector = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=40
        )

        st.info("🔄 Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            height, width, _ = frame.shape

            # Apply region of interest for highway demo, otherwise process full frame
            if video_source == "Demo Video (highway.mp4)" and height >= 720 and width >= 800:
                y1, y2, x1, x2 = 340, 720, 500, 800
                roi = frame[y1:y2, x1:x2]
            else:
                y1, y2, x1, x2 = 0, height, 0, width
                roi = frame

            # 1. Object Detection via MOG2
            mask = object_detector.apply(roi)
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            detections = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append([x, y, w, h])

            # 2. Object Tracking
            boxes_ids = tracker.update(detections)
            display_frame = frame.copy()

            # Draw ROI boundary if demo video
            if video_source == "Demo Video (highway.mp4)":
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 120, 0), 2)

            # Draw tracking bounding boxes and ID tags
            for box_id in boxes_ids:
                x, y, w, h, obj_id = box_id
                real_x = x + x1
                real_y = y + y1

                # Bounding box
                cv2.rectangle(display_frame, (real_x, real_y), (real_x + w, real_y + h), (0, 255, 0), 2)
                # ID label
                cv2.putText(
                    display_frame,
                    f"ID: {obj_id}",
                    (real_x, max(20, real_y - 8)),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.8,
                    (255, 0, 0),
                    2
                )

            # Convert BGR to RGB for Streamlit rendering
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                frame_rgb,
                channels="RGB",
                caption="Tracked Video Feed",
                use_container_width=True
            )

            mask_placeholder.image(
                mask,
                caption="Motion Mask (MOG2)",
                use_container_width=True
            )

        cap.release()
        st.success("✅ Video processing completed!")

else:
    st.info("Please select or upload a video file to begin.")
