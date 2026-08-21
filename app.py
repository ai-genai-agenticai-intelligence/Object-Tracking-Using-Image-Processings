import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from tracker import EuclideanDistTracker

st.set_page_config(page_title="Object Tracking & Analytics", layout="wide")

st.title("🚗 Moving Object Tracking & Analytics")
st.write("Detect moving objects with Background Subtraction (MOG2) and track them with persistent IDs and telemetry.")

# Video Source Selection
source_option = st.radio(
    "Select Video Source:",
    ["Demo Video (highway.mp4)", "Upload Custom Video"],
    horizontal=True
)

video_path = None

if source_option == "Demo Video (highway.mp4)":
    if os.path.exists("highway.mp4"):
        video_path = "highway.mp4"
    else:
        st.error("Demo video 'highway.mp4' not found.")
else:
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()
        video_path = tfile.name

if video_path is not None:
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = EuclideanDistTracker()
    detector = cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=40
    )

    # Top Live Metrics Bar
    m_col1, m_col2, m_col3 = st.columns(3)
    metric_total = m_col1.empty()
    metric_active = m_col2.empty()
    metric_frames = m_col3.empty()

    metric_total.metric("🏷️ Total Objects Counted", "0")
    metric_active.metric("🎯 Active in Current Frame", "0")
    metric_frames.metric("🎞️ Frame Progress", f"0 / {total_video_frames if total_video_frames > 0 else 'N/A'}")

    # Video & Mask Columns
    frame_box, mask_box = st.columns(2)
    video_frame = frame_box.empty()
    mask_frame = mask_box.empty()

    # Active Tracking Details Telemetry Table
    st.subheader("📊 Live Object Telemetry & Details")
    table_placeholder = st.empty()

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        height, width, _ = frame.shape

        # Highway Demo ROI or full frame
        if source_option == "Demo Video (highway.mp4)" and height >= 720 and width >= 800:
            y1, y2, x1, x2 = 340, 720, 500, 800
            roi = frame[y1:y2, x1:x2]
        else:
            y1, y2, x1, x2 = 0, height, 0, width
            roi = frame

        # 1. Motion Detection with MOG2
        mask = detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        # 2. Object Tracking
        boxes_ids = tracker.update(detections)

        # Prepare active telemetry details
        active_telemetry = []

        # Draw ROI Boundary if demo video
        if source_option == "Demo Video (highway.mp4)":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)

        for x, y, w, h, obj_id in boxes_ids:
            real_x = x + x1
            real_y = y + y1
            cx = real_x + w // 2
            cy = real_y + h // 2

            active_telemetry.append({
                "Object ID": f"ID {obj_id}",
                "Centroid (X, Y)": f"({cx}, {cy})",
                "Bounding Box (W x H)": f"{w} x {h} px",
                "Est. Area": f"{w * h} px²"
            })

            # Draw Bounding Box
            cv2.rectangle(frame, (real_x, real_y), (real_x + w, real_y + h), (0, 255, 0), 2)

            # Draw Centroid
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Draw ID Tag with background for readability
            label = f"ID: {obj_id}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (real_x, max(0, real_y - text_h - 8)), (real_x + text_w + 8, real_y), (0, 255, 0), -1)
            cv2.putText(
                frame,
                label,
                (real_x + 4, max(text_h, real_y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        # On-screen HUD banner
        hud_text = f"Active: {len(boxes_ids)} | Total Counted: {tracker.id_count}"
        cv2.putText(
            frame,
            hud_text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # Update Live Metrics
        metric_total.metric("🏷️ Total Objects Counted", f"{tracker.id_count}")
        metric_active.metric("🎯 Active in Current Frame", f"{len(boxes_ids)}")
        metric_frames.metric("🎞️ Frame Progress", f"{frame_count} / {total_video_frames if total_video_frames > 0 else 'N/A'}")

        # Update Live Telemetry Table
        if active_telemetry:
            table_placeholder.dataframe(pd.DataFrame(active_telemetry), use_container_width=True)
        else:
            table_placeholder.info("No active objects in the current frame.")

        # Render Video & Mask
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame.image(frame_rgb, caption=f"Tracked Video Feed (Frame {frame_count})", use_container_width=True)
        mask_frame.image(mask, caption="MOG2 Motion Mask", use_container_width=True)

    cap.release()
    st.success(f"✅ Tracking completed! Total unique objects tracked: **{tracker.id_count}**")

else:
    st.info("Please select or upload a video file to begin tracking.")
