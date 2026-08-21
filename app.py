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
    "and **Euclidean Distance Centroid Tracking**."
)

# Sidebar Settings
st.sidebar.header("⚙️ Configuration & Source")

video_source = st.sidebar.radio(
    "Select Video Source",
    options=["Demo Video (highway.mp4)", "Upload Custom Video"],
    index=0
)

video_path = None

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
        # Save uploaded file persistently in tempdir using its unique name
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f"cv_upload_{uploaded_file.name}")
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getvalue())
        video_path = temp_filename

# Algorithm Parameters
st.sidebar.subheader("🔬 Detection Parameters")
history = st.sidebar.slider(
    "MOG2 History",
    min_value=30,
    max_value=500,
    value=100,
    step=10,
    help="Number of previous frames used for background modeling."
)
var_threshold = st.sidebar.slider(
    "Variance Threshold",
    min_value=10,
    max_value=120,
    value=40,
    step=5,
    help="Threshold on the squared Mahalanobis distance. Higher values reduce false positives."
)
min_area = st.sidebar.slider(
    "Min Contour Area (px²)",
    min_value=20,
    max_value=2000,
    value=100,
    step=20,
    help="Filter out small noise contours below this area."
)
apply_threshold = st.sidebar.checkbox(
    "Apply Binary Clean Threshold (254)",
    value=True,
    help="Removes gray shadow artifacts from MOG2 mask."
)

# ROI settings
st.sidebar.subheader("🎯 Region of Interest (ROI)")
roi_mode = st.sidebar.selectbox(
    "ROI Mode",
    options=["Full Frame", "Highway ROI (340:720, 500:800)", "Custom ROI Sliders"],
    index=1 if video_source == "Demo Video (highway.mp4)" else 0,
    help="Full Frame processes the entire video. Highway ROI is tailored for highway.mp4."
)

# Custom ROI sliders if selected
y_start, y_end, x_start, x_end = 0, 100, 0, 100
if roi_mode == "Custom ROI Sliders":
    st.sidebar.caption("Percentage of frame height & width to process:")
    y_start, y_end = st.sidebar.slider("Height Range (%)", 0, 100, (30, 100))
    x_start, x_end = st.sidebar.slider("Width Range (%)", 0, 100, (20, 90))

# Playback Speed Control
fps_limit = st.sidebar.slider("Playback Speed (FPS limit)", min_value=5, max_value=60, value=30, step=5)

# Main Application Logic
if video_path is None:
    st.info("👆 Please select or upload a video file from the sidebar to begin tracking.")
else:
    # Probe video metadata
    cap_probe = cv2.VideoCapture(video_path)
    if not cap_probe.isOpened():
        st.error("❌ Could not read the selected video file. Please check the file format.")
    else:
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap_probe.get(cv2.CAP_PROP_FPS)
        ret_first, first_frame = cap_probe.read()
        cap_probe.release()

        # Display Video Info Banner
        st.caption(f"📁 Video loaded: `{os.path.basename(video_path)}` | Resolution: `{orig_w}x{orig_h}` | Total Frames: `{total_frames}` | Source FPS: `{video_fps:.1f}`")

        # Session State for tracking control
        if "is_running" not in st.session_state:
            st.session_state.is_running = False

        col_btn1, col_btn2, _ = st.columns([1.2, 1.2, 3.6])
        with col_btn1:
            if st.button("▶ Start Tracking", type="primary", use_container_width=True):
                st.session_state.is_running = True
        with col_btn2:
            if st.button("⏹ Stop Tracking", type="secondary", use_container_width=True):
                st.session_state.is_running = False

        # Live Metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        metric_total = m_col1.empty()
        metric_active = m_col2.empty()
        metric_frames = m_col3.empty()
        metric_fps = m_col4.empty()

        metric_total.metric("Total Tracked IDs", "0")
        metric_active.metric("Active Objects", "0")
        metric_frames.metric("Frame", f"0 / {total_frames}")
        metric_fps.metric("Processing FPS", "0.0")

        # Visual Feed Columns
        col_video, col_mask = st.columns(2)
        with col_video:
            st.subheader("📹 Video Tracking Feed")
            frame_placeholder = st.empty()
        with col_mask:
            st.subheader("🎭 Foreground Mask (MOG2)")
            mask_placeholder = st.empty()

        # Show initial preview if not running yet
        if not st.session_state.is_running and ret_first and first_frame is not None:
            preview_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(preview_rgb, caption="Video Preview (Ready to Start)", use_container_width=True)
            mask_placeholder.info("Click '▶ Start Tracking' above to begin real-time detection & tracking.")

        # Processing Loop
        if st.session_state.is_running:
            cap = cv2.VideoCapture(video_path)
            tracker = EuclideanDistTracker()
            object_detector = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold
            )

            frame_idx = 0
            start_time = time.time()
            progress_bar = st.progress(0)

            while cap.isOpened() and st.session_state.is_running:
                loop_start = time.time()
                ret, frame = cap.read()

                if not ret or frame is None:
                    st.success("🎉 Video completed successfully!")
                    st.session_state.is_running = False
                    break

                frame_idx += 1
                fh, fw, _ = frame.shape

                # Determine ROI coordinates
                if roi_mode == "Highway ROI (340:720, 500:800)" and fh >= 720 and fw >= 800:
                    y1, y2, x1, x2 = 340, 720, 500, 800
                elif roi_mode == "Custom ROI Sliders":
                    y1 = int(fh * y_start / 100.0)
                    y2 = int(fh * y_end / 100.0)
                    x1 = int(fw * x_start / 100.0)
                    x2 = int(fw * x_end / 100.0)
                    # Safety check
                    if y2 <= y1 or x2 <= x1:
                        y1, y2, x1, x2 = 0, fh, 0, fw
                else:
                    # Full Frame
                    y1, y2, x1, x2 = 0, fh, 0, fw

                roi = frame[y1:y2, x1:x2]

                # 1. Detection via MOG2
                mask = object_detector.apply(roi)
                if apply_threshold:
                    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                detections = []

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > min_area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        detections.append([x, y, w, h])

                # 2. Object Tracking
                boxes_ids = tracker.update(detections)

                # 3. Draw Bounding Boxes and ID Tags on Display Frame
                display_frame = frame.copy()

                # Highlight ROI boundaries if not Full Frame
                if roi_mode != "Full Frame":
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
                    cv2.putText(
                        display_frame,
                        "ROI Active",
                        (x1 + 5, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 120, 0),
                        2
                    )

                for box_id in boxes_ids:
                    x, y, w, h, obj_id = box_id
                    real_x = x + x1
                    real_y = y + y1

                    # Draw bounding box
                    cv2.rectangle(display_frame, (real_x, real_y), (real_x + w, real_y + h), (0, 255, 0), 2)

                    # Draw ID badge
                    label = f"ID: {obj_id}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(
                        display_frame,
                        (real_x, max(0, real_y - lh - 10)),
                        (real_x + lw + 10, real_y),
                        (0, 255, 0),
                        -1
                    )
                    cv2.putText(
                        display_frame,
                        label,
                        (real_x + 5, max(15, real_y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2
                    )

                # Performance and Stats
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0.0

                # Update live metrics
                metric_total.metric("Total Tracked IDs", f"{tracker.id_count}")
                metric_active.metric("Active Objects", f"{len(boxes_ids)}")
                metric_frames.metric("Frame", f"{frame_idx} / {total_frames}")
                metric_fps.metric("Processing FPS", f"{current_fps:.1f}")

                if total_frames > 0:
                    progress_bar.progress(min(1.0, frame_idx / total_frames))

                # Render video & mask to placeholders
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, caption=f"Frame {frame_idx}", use_container_width=True)
                mask_placeholder.image(mask, caption="MOG2 Motion Mask", use_container_width=True)

                # Dynamic delay to match target FPS
                target_frame_time = 1.0 / fps_limit
                process_time = time.time() - loop_start
                if target_frame_time > process_time:
                    time.sleep(target_frame_time - process_time)

            cap.release()