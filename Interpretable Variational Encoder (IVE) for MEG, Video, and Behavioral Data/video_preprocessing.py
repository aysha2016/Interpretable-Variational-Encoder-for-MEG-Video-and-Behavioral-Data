import cv2
import numpy as np
import os

def preprocess_video(video_path, output_folder, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // frame_rate)
    frame_count = 0
    saved_frames = []

    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            frame_file = os.path.join(output_folder, f"frame_{frame_count}.npy")
            np.save(frame_file, frame_resized)
            saved_frames.append(frame_file)
        frame_count += 1

    cap.release()
    return saved_frames