import cv2
import os


def extract_frames(video_path, output_folder, num_frames=16):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(total_frames // num_frames, 1)

    frames = []

    for i in range(num_frames):

        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)

        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{i}.jpg")

        cv2.imwrite(frame_path, frame)

        frames.append(frame_path)

    cap.release()

    return frames