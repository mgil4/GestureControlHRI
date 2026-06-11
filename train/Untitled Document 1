import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

#!/usr/bin/env python3
"""
Step 1: Collect hand gesture data (ROS2 version)
------------------------------------------------
Uses /camera/camera/color/image_raw instead of cv2.VideoCapture.

Run:
    python3 collect_gestures_ros.py
"""

import cv2
import csv
import os
import time
import urllib.request

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image, ImageFormat


# =========================
# CONFIG
# =========================
OUTPUT_CSV     = "gesture_data_luka.csv"
SAMPLES_NEEDED = 20
COLLECT_DELAY  = 0.05

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


# =========================
# MODEL DOWNLOAD
# =========================
def download_model(path, url):
    if not os.path.exists(path):
        print(f"[INFO] Downloading model → {path}")
        urllib.request.urlretrieve(url, path)


# =========================
# NORMALIZATION
# =========================
def normalize_landmarks(landmarks, w, h):
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    bx, by = coords[0]
    rel = [(x - bx, y - by) for x, y in coords]
    flat = [v for p in rel for v in p]
    mx = max(abs(v) for v in flat) or 1
    return [v / mx for v in flat]


# =========================
# DRAW HAND
# =========================
def draw_hand(frame, landmarks, w, h):
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, coords[a], coords[b], (200, 200, 200), 1)

    for x, y in coords:
        cv2.circle(frame, (x, y), 5, (0, 220, 100), -1)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)


# =========================
# ROS CAMERA NODE
# =========================
class CameraNode(Node):
    def __init__(self):
        super().__init__('gesture_collector')
        self.bridge = CvBridge()
        self.frame = None

        self.create_subscription(
            RosImage,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


# =========================
# MAIN
# =========================
def main():
    download_model(LANDMARKER_MODEL, LANDMARKER_URL)

    # -------------------------
    # ROS init
    # -------------------------
    rclpy.init()
    node = CameraNode()

    # -------------------------
    # gesture name
    # -------------------------
    print("\n" + "="*50)
    print("  GESTURE DATA COLLECTOR (ROS2)")
    print("="*50)

    gesture_name = input("Enter gesture name: ").strip()
    if not gesture_name:
        print("No name entered.")
        return

    # -------------------------
    # CSV setup
    # -------------------------
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
        writer.writerow(header)

    # count existing samples
    existing = 0
    if file_exists:
        with open(OUTPUT_CSV) as f:
            for row in csv.reader(f):
                if row and row[0] == gesture_name:
                    existing += 1

    print(f"[INFO] Existing samples: {existing}")
    print(f"[INFO] Need: {SAMPLES_NEEDED}")

    # -------------------------
    # MediaPipe setup
    # -------------------------
    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # -------------------------
    # runtime loop
    # -------------------------
    collected = 0
    last_capture = 0

    while collected < SAMPLES_NEEDED:
        rclpy.spin_once(node, timeout_sec=0.01)

        frame = node.frame
        if frame is None:
            continue

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        hand_detected = bool(result.hand_landmarks)

        if hand_detected:
            draw_hand(frame, result.hand_landmarks[0], w, h)

        # UI
        progress = collected / SAMPLES_NEEDED
        bar_w = int(w * 0.6)
        bar_x = int(w * 0.2)
        bar_y = h - 50
        filled = int(bar_w * progress)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 18), (0, 200, 80), -1)

        status = "Hand detected" if hand_detected else "No hand detected"
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0,220,100) if hand_detected else (0,0,255), 2)
        cv2.putText(frame, f"{collected}/{SAMPLES_NEEDED}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        cv2.imshow("ROS Gesture Collector", frame)

        key = cv2.waitKey(10) & 0xFF
        now = time.time()

        if key in (27, ord('q')):
            break

        # SPACE capture
        if key == 32 and hand_detected and (now - last_capture) >= COLLECT_DELAY:
            features = normalize_landmarks(result.hand_landmarks[0], w, h)
            writer.writerow([gesture_name] + features)
            csv_file.flush()

            collected += 1
            last_capture = now

    # -------------------------
    # cleanup
    # -------------------------
    csv_file.close()
    landmarker.close()
    node.destroy_node()
    rclpy.shutdown()

    cv2.destroyAllWindows()

    print(f"\nDone. Collected {collected} samples for '{gesture_name}'.")


if __name__ == "__main__":
    main()


