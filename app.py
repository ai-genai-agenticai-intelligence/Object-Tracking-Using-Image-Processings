import streamlit as st
import cv2
import tempfile
from tracker import EuclideanDistTracker

st.set_page_config(page_title="Object Tracking", layout="wide")

st.title("🚗 Moving Object Tracking with IDs")

video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if video:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    cap = cv2.VideoCapture(tfile.name)

    tracker = EuclideanDistTracker()

    detector = cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=40
    )

    # Live Tracking Details Metrics
    col_m1, col_m2 = st.columns(2)
    metric_total = col_m1.empty()
    metric_active = col_m2.empty()

    frame_box, mask_box = st.columns(2)

    video_frame = frame_box.empty()
    mask_frame = mask_box.empty()

    # Tracking Details Box
    details_box = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        roi = frame

        # Detect motion
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

            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        # Track objects
        boxes_ids = tracker.update(detections)

        active_ids = []

        for x, y, w, h, obj_id in boxes_ids:
            active_ids.append(f"ID {obj_id}")

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw ID label
            cv2.putText(
                frame,
                f"ID {obj_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # On-screen HUD details
        cv2.putText(
            frame,
            f"Active: {len(boxes_ids)} | Total: {tracker.id_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # Update Live Details & Metrics
        metric_total.metric("🏷️ Total Objects Counted", f"{tracker.id_count}")
        metric_active.metric("🎯 Active in Current Frame", f"{len(boxes_ids)}")

        if active_ids:
            details_box.info(f"📋 **Active Tracked Objects:** {', '.join(active_ids)}")
        else:
            details_box.caption("No moving objects detected in this frame.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_frame.image(frame, use_container_width=True)
        mask_frame.image(mask, use_container_width=True)

    cap.release()
    st.success(f"✅ Tracking Finished! Total unique objects detected: **{tracker.id_count}**")

else:
    st.info("Please upload a video file to begin tracking.")
