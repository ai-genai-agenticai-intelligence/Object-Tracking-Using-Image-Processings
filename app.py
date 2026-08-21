import streamlit as st
import cv2
import tempfile
from tracker import EuclideanDistTracker

st.set_page_config(page_title="Object Tracking", layout="wide")

st.title("🚗 Moving Object Tracking with Image Preprocessing")

video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

if video:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    cap = cv2.VideoCapture(tfile.name)

    tracker = EuclideanDistTracker()

    detector = cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=40
    )

    frame_box, mask_box = st.columns(2)

    video_frame = frame_box.empty()
    mask_frame = mask_box.empty()

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

        for x, y, w, h, obj_id in boxes_ids:

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(
                frame,
                f"ID {obj_id}",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_frame.image(frame, use_container_width=True)
        mask_frame.image(mask, use_container_width=True)

    cap.release()
    st.success("Tracking Finished")
