"""
Collect hand gesture data — ROS 2 camera version
--------------------------------------------------
Subscribes to a ROS 2 image topic instead of opening a local webcam.
Hold a gesture in front of the camera and press SPACE to record samples.
Repeat for each gesture you want to train.

Run (source your workspace first):
    python collect_gestures_ros.py

    # Override the camera topic if needed:
    python collect_gestures_ros.py --ros-args -p image_topic:=/head_front_camera/color/image_raw

Output:
    gesture_data.csv   — one row per sample (label + 42 landmark values)
"""

import csv
import os
import sys
import time
import threading
import urllib.request

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as MpImage, ImageFormat

# ── Config ────────────────────────────────────────────────────────────────────

IMAGE_TOPIC      = "/camera/camera/color/image_raw"
OUTPUT_CSV       = "gesture_data.csv"
SAMPLES_NEEDED   = 10
COLLECT_DELAY    = 0.05    # seconds between auto-captures while SPACE is held

LANDMARKER_MODEL = "hand_landmarker.task"
LANDMARKER_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_model(path, url):
    if not os.path.exists(path):
        print(f"[INFO] Downloading model → {path} ...")
        urllib.request.urlretrieve(url, path)
        print("[INFO] Done.")


def normalize_landmarks(landmarks, w, h):
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    bx, by = coords[0]
    rel    = [(x - bx, y - by) for x, y in coords]
    flat   = [v for pair in rel for v in pair]
    mx     = max(abs(v) for v in flat) or 1
    return [v / mx for v in flat]


def draw_hand(frame, landmarks, w, h):
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, coords[a], coords[b], (200, 200, 200), 1, cv2.LINE_AA)
    for x, y in coords:
        cv2.circle(frame, (x, y), 5, (0, 220, 100), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (0, 0, 0),      1, cv2.LINE_AA)


def ros_image_to_numpy(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image to BGR numpy array without cv_bridge."""
    enc = msg.encoding.lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ('rgb8', 'bgr8'):
        frame = raw.reshape((msg.height, msg.width, 3))
        if enc == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif enc == 'mono8':
        frame = raw.reshape((msg.height, msg.width))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif enc == '16uc1':
        frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            (msg.height, msg.width)
        )
        frame = (frame / 256).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif enc == '32fc1':
        frame = np.frombuffer(msg.data, dtype=np.float32).reshape(
            (msg.height, msg.width)
        )
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f'Unsupported image encoding: {msg.encoding}')

    return frame


# ── ROS 2 image subscriber ────────────────────────────────────────────────────

class ImageReceiver(Node):
    """
    Minimal ROS 2 node that subscribes to a camera topic and stores
    the latest frame in a thread-safe slot for the main loop to read.
    """

    def __init__(self, topic: str):
        super().__init__("gesture_collector")
        self._lock        = threading.Lock()
        self._latest      = None   # most recent BGR numpy frame

        self.create_subscription(Image, topic, self._cb, 10)
        self.get_logger().info(f"Subscribed to {topic}")

    def _cb(self, msg: Image):
        try:
            frame = ros_image_to_numpy(msg)
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return
        with self._lock:
            self._latest = frame

    def get_frame(self):
        """Returns a copy of the latest frame, or None if none received yet."""
        with self._lock:
            return self._latest.copy() if self._latest is not None else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    download_model(LANDMARKER_MODEL, LANDMARKER_URL)

    # Ask what gesture to record before spinning up ROS
    print("\n" + "="*50)
    print("  GESTURE DATA COLLECTOR  —  ROS 2 camera")
    print("="*50)
    gesture_name = input(
        "Enter gesture name (e.g. 'thumbs_left', 'peace', 'stop'): "
    ).strip()
    if not gesture_name:
        print("No name entered. Exiting.")
        return

    # ── CSV setup ─────────────────────────────────────────────────────────
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file    = open(OUTPUT_CSV, "a", newline="")
    writer      = csv.writer(csv_file)
    if not file_exists:
        header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
        writer.writerow(header)

    existing = 0
    if file_exists:
        with open(OUTPUT_CSV) as f:
            for row in csv.reader(f):
                if row and row[0] == gesture_name:
                    existing += 1
    print(f"\n[INFO] Existing samples for '{gesture_name}': {existing}")
    print(f"[INFO] Will collect {SAMPLES_NEEDED} more.")
    print("\nInstructions:")
    print("  - Hold your gesture in front of the camera")
    print("  - Press and HOLD SPACE to record samples")
    print("  - Press Q or ESC to stop early\n")

    # ── MediaPipe setup ────────────────────────────────────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL)
    options      = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # ── ROS 2 setup ────────────────────────────────────────────────────────
    rclpy.init()
    receiver = ImageReceiver(IMAGE_TOPIC)

    # Spin ROS in a background thread so the main thread can run the
    # OpenCV display loop (cv2.imshow must run on the main thread)
    spin_thread = threading.Thread(target=rclpy.spin, args=(receiver,), daemon=True)
    spin_thread.start()

    print("[INFO] Waiting for first camera frame ...")
    while True:
        if receiver.get_frame() is not None:
            break
        time.sleep(0.05)
    print("[INFO] Camera feed received. Starting collection.\n")

    # ── Collection loop ────────────────────────────────────────────────────
    collected    = 0
    last_capture = 0

    while collected < SAMPLES_NEEDED:
        frame = receiver.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        hand_detected = bool(result.hand_landmarks)

        if hand_detected:
            draw_hand(frame, result.hand_landmarks[0], w, h)

        # ── UI overlay ─────────────────────────────────────────────────────
        progress = collected / SAMPLES_NEEDED
        bar_w    = int(w * 0.6)
        bar_x    = int(w * 0.2)
        bar_y    = h - 50
        filled   = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + 18), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled, bar_y + 18), (0, 200, 80), -1)

        status_color = (0, 220, 100) if hand_detected else (50, 50, 220)
        status_text  = (
            "Hand detected — hold SPACE to record"
            if hand_detected else
            "No hand detected"
        )
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, status_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Samples: {collected} / {SAMPLES_NEEDED}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Topic: {IMAGE_TOPIC}", (10, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow(f"Collecting: {gesture_name}  |  Q/ESC = quit", frame)

        key = cv2.waitKey(10) & 0xFF
        now = time.time()

        if key in (ord("q"), ord("Q"), 27):
            print("\n[INFO] Stopped early by user.")
            break

        # SPACE held: record if hand visible and cooldown elapsed
        if key == 32 and hand_detected and (now - last_capture) >= COLLECT_DELAY:
            features     = normalize_landmarks(result.hand_landmarks[0], w, h)
            writer.writerow([gesture_name] + features)
            csv_file.flush()
            collected   += 1
            last_capture = now
            print(f"  Sample {collected}/{SAMPLES_NEEDED} recorded", end="\r")

    # ── Cleanup ────────────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    landmarker.close()
    csv_file.close()
    receiver.destroy_node()
    rclpy.shutdown()

    print(f"\n\nCollected {collected} samples for '{gesture_name}'.")
    print(f"[INFO] Saved to {OUTPUT_CSV}")
    print(f"[INFO] Run this script again for your next gesture.")
    print(f"[INFO] When all gestures are done, run train_classifier.py")


if __name__ == "__main__":
    main()
