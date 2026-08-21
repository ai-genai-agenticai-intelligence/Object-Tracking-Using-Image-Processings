import streamlit as st
import cv2
import tempfile
import os
import time
from tracker import EuclideanDistTracker

# Page Configuration
st.set_page_config(
    page_title="Object Tracking AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🚗 Real-Time Vehicle Tracking & Detection")
st.markdown(
    "Computer vision pipeline using **Gaussian Mixture Background Subtraction (MOG2)** "
    "and **Euclidean Distance Centroid Tracking** with OpenCV."
)

# Sidebar Settings
st.sidebar.header("⚙️ Configuration & Settings")

# Video Source Selection
video_source = st.sidebar.radio(
    "Select Video Source",
    options=["Demo Video (highway.mp4)", "Upload Custom Video"],
    index=0
)

video_path = None
temp_file_path = None

if video_source == "Demo Video (highway.mp4)":
    default_video = "highway.mp4"
    if os.path.exists(default_video):
        video_path = default_video
    else:
        st.sidebar.error(f"Demo file '{default_video}' not found in root directory.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Video File",
        type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        temp_file_path = tfile.name
        video_path = temp_file_path

# Algorithm Parameters
st.sidebar.subheader("🔬 Detection Parameters")
history = st.sidebar.slider("MOG2 History", min_value=50, max_value=500, value=100, step=10, help="Number of frames used for background modeling.")
var_threshold = st.sidebar.slider("Variance Threshold", min_value=10, max_value=100, value=40, step=5, help="Threshold for foreground pixel classification.")
min_area = st.sidebar.slider("Min Contour Area (px²)", min_value=50, max_value=1000, value=150, step=25, help="Filter out small noise contours below this area.")
apply_threshold = st.sidebar.checkbox("Apply Binary Clean Threshold (254)", value=True, help="Removes gray shadow artifacts from MOG2 mask.")

# ROI settings
st.sidebar.subheader("🎯 Region of Interest (ROI)")
use_roi = st.sidebar.checkbox("Use ROI Filter (Optimized for Highway)", value=True, help="Crops region of interest to eliminate irrelevant background noise.")

# Playback Speed
fps_limit = st.sidebar.slider("Target FPS / Playback Delay (ms)", min_value=1, max_value=60, value=25, step=1)

# Main Dashboard Layout
if video_path is None:
    st.info("👆 Please select or upload a video from the sidebar to begin.")
else:
    # State for Start / Stop
    if "tracking_active" not in st.session_state:
        st.session_state.tracking_active = False

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("▶ Start Tracking", type="primary", use_container_width=True):
            st.session_state.tracking_active = True
    with col_btn2:
        if st.button("⏹ Stop Tracking", type="secondary", use_container_width=True):
            st.session_state.tracking_active = False

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    metric_total = m_col1.empty()
    metric_active = m_col2.empty()
    metric_frames = m_col3.empty()
    metric_fps = m_col4.empty()

    metric_total.metric("Total Tracked IDs", "0")
    metric_active.metric("Active Objects", "0")
    metric_frames.metric("Frame Number", "0")
    metric_fps.metric("FPS", "0.0")

    # Display Columns
    col_video, col_mask = st.columns(2)
    with col_video:
        st.subheader("📹 Tracking Feed")
        frame_placeholder = st.empty()
    with col_mask:
        st.subheader("🎭 Motion Mask")
        mask_placeholder = st.empty()

    if st.session_state.tracking_active:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Error: Could not open video stream.")
        else:
            # Initialize Tracker and Detector
            tracker = EuclideanDistTracker()
            object_detector = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold
            )

            frame_count = 0
            start_time = time.time()

            while cap.isOpened() and st.session_state.tracking_active:
                ret, frame = cap.read()
                if not ret:
                    st.success("🎉 Video processing complete!")
                    st.session_state.tracking_active = False
                    break

                frame_count += 1
                h, w, _ = frame.shape

                # Apply ROI if enabled
                if use_roi and h >= 720 and w >= 800:
                    roi = frame[340:720, 500:800]
                    offset_x, offset_y = 500, 340
                else:
                    roi = frame
                    offset_x, offset_y = 0, 0

                # 1. Detection
                mask = object_detector.apply(roi)
                if apply_threshold:
                    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                detections = []

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > min_area:
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        detections.append([x, y, cw, ch])

                # 2. Tracking
                boxes_ids = tracker.update(detections)

                # Draw bounding boxes & IDs on frame
                display_frame = frame.copy()

                # Highlight ROI rectangle if enabled
                if use_roi and (offset_x > 0 or offset_y > 0):
                    cv2.rectangle(display_frame, (500, 340), (800, 720), (255, 100, 0), 2)
                    cv2.putText(display_frame, "ROI Area", (505, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

                for box_id in boxes_ids:
                    x, y, bw, bh, obj_id = box_id
                    real_x = x + offset_x
                    real_y = y + offset_y

                    # Draw Bounding Box & ID Label
                    cv2.rectangle(display_frame, (real_x, real_y), (real_x + bw, real_y + bh), (0, 255, 0), 2)
                    cv2.rectangle(display_frame, (real_x, real_y - 25), (real_x + 65, real_y), (0, 255, 0), -1)
                    cv2.putText(
                        display_frame,
                        f"ID {obj_id}",
                        (real_x + 5, real_y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2
                    )

                # Performance calculation
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0.0

                # Update Metrics
                metric_total.metric("Total Tracked IDs", f"{tracker.id_count}")
                metric_active.metric("Active Objects", f"{len(boxes_ids)}")
                metric_frames.metric("Frame Number", f"{frame_count}")
                metric_fps.metric("FPS", f"{current_fps:.1f}")

                # Update Streamlit image placeholders
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                mask_placeholder.image(mask, channels="GRAY", use_container_width=True)

                # Delay control
                time.sleep(1.0 / fps_limit)

            cap.release()

    # Clean up temporary uploaded file on exit
    if temp_file_path and os.path.exists(temp_file_path) and not st.session_state.tracking_active:
        try:
            os.remove(temp_file_path)
        except Exception:
            pass