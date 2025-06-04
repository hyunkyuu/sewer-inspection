import os
import cv2
import csv
import math
import shutil
from datetime import timedelta


def extract_frames(video_path, output_dir, seconds_interval=0.5, reset=False):

    if reset and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = math.ceil(fps * seconds_interval)

    frame_count = 0
    saved_count = 0

    timestamp_csv = os.path.join(output_dir, "timestamps.csv")
    with open(timestamp_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_filename", "frame_number", "timestamp"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                # frame
                frame_filename = f"frame_{saved_count:05d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                # timestamp
                time_seconds = frame_count / fps
                timestamp = str(timedelta(seconds=time_seconds))
                writer.writerow([frame_filename, frame_count, timestamp])

                saved_count += 1

            frame_count += 1

    cap.release()
    print(f"saved {saved_count} frames including timestamps on {output_dir}")


if __name__ == "__main__":
    video_file = "video/raw/sample.mp4"
    output_folder = "video/frames"
    extract_frames(video_file, output_folder, seconds_interval=0.5, reset=True)
