#!/usr/bin/env python3
"""
hand_gesture_recognition.py

Responsibilities
  • Subscribe to a camera image topic (sensor_msgs/Image)
  • Detect hand landmarks with MediaPipe HandLandmarker
  • Classify each hand with a pre-trained sklearn classifier
  • Accumulate per-frame predictions in a sliding time window
  • Publish the majority-vote label + aggregated confidence on /gesture/result
    as a JSON string inside a std_msgs/String message

This node is COMPLETELY MODE-AGNOSTIC.  It knows nothing about robot commands,
text building, or LLM interaction.  It only publishes reliable gesture labels.

Published topics:
  /gesture/result  (std_msgs/String)
      JSON payload:
      {
        "label":      "A",          # majority label in the voting window
        "confidence": 0.87,         # mean confidence of winning predictions
        "num_hands":  1,            # number of hands visible in latest frame
        "hands": [                  # one entry per visible hand (latest frame)
            {"label": "A", "confidence": 0.92, "hand_index": 0}
        ],
        "timestamp":  1234567890.1  # seconds (float)
      }

Subscribed topics:
  /<image_topic>  (sensor_msgs/Image)   default: /image_raw

Install dependencies:
  pip install opencv-python mediapipe scikit-learn numpy

Usage (standalone launch):
  ros2 run <your_pkg> hand_gesture_recognition \
      --ros-args -p model_path:=/abs/path/gesture_classifier.pkl \
                 -p landmarker_model:=/abs/path/hand_landmarker.task \
                 -p image_topic:=image_raw \
                 -p voting_window_sec:=0.75
"""

import json
import os
import pickle
import time
from collections import deque, Counter

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
#from rclpy.parameter import Parameter
#from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange, IntegerRange
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as MpImage, ImageFormat
from ament_index_python.packages import get_package_share_directory

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# Helper functions
def normalize_landmarks(hand_landmarks, img_w: int, img_h: int):
    """
    Convert MediaPipe landmark list to a normalised feature vector:
    1. Convert to pixel coordinates.
    2. Make wrist (landmark 0) the origin.
    3. Flatten to a 1-D array and scale to [-1, 1] by the max absolute value.
    Returns:
    features : np.ndarray  shape (42,)  – ready for classifier
    pixel_coords : list[tuple[int,int]]  – for optional overlay drawing
    """
    coords = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in hand_landmarks]

    wrist_x, wrist_y = coords[0]
    relative = [(x - wrist_x, y - wrist_y) for x, y in coords]

    flat = [v for pair in relative for v in pair]
    scale = max(abs(v) for v in flat) or 1.0

    features = np.array([v / scale for v in flat], dtype=np.float32)
    return features, coords

def majority_vote(window: deque):
    """
    Given a deque of (label, confidence, timestamp) tuples,
    return (winning_label, mean_confidence_of_winner) or (None, 0.0).
    """
    if not window:
        return None, 0.0

    label_counts = Counter(entry[0] for entry in window)
    winner = label_counts.most_common(1)[0][0]

    winner_confidences = [entry[1] for entry in window if entry[0] == winner]
    mean_conf = float(np.mean(winner_confidences))

    return winner, mean_conf

# ROS 2 Node
class HandGestureRecognitionNode(Node):
    """
    Subscribes to camera images, runs MediaPipe + ML classifier per frame,
    accumulates predictions in a sliding window, and periodically publishes
    the majority-voted gesture label with aggregated confidence.
    """
    def __init__(self):
        super().__init__('hand_gesture_recognition')

        # Declare parameters
        self._declare_params()

        # Read parameters
        self.image_topic         = self.get_parameter('image_topic').value
        self.model_path          = self.get_parameter('model_path').value
        self.landmarker_model    = self.get_parameter('landmarker_model').value
        self.min_confidence      = self.get_parameter('min_confidence').value
        self.num_hands           = self.get_parameter('num_hands').value
        self.voting_window_sec   = self.get_parameter('voting_window_sec').value
        self.publish_rate_hz     = self.get_parameter('publish_rate_hz').value
        self.detect_conf         = self.get_parameter('detection_confidence').value
        self.track_conf          = self.get_parameter('tracking_confidence').value
        self.required_stability  = self.get_parameter('required_stability').value
        self.show_preview        = self.get_parameter('show_preview').value
        
        # Load models
        self._load_classifier()
        self._init_landmarker()

        # Sliding voting window
        self._vote_window: deque = deque()

        # Latest raw per-hand results
        self._latest_hands: list = []
        self._last_label = None
        self._stable_count = 0

        self._pub = self.create_publisher(String, '/gesture/result', 10)

        qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT, # or BEST_EFFORT (try both)
        history=HistoryPolicy.KEEP_LAST,
        depth=10
        )
        
        self._image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile=qos_profile_sensor_data,
        )

        # Timer drives the publish cycle independently of camera FPS
        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self._publish_timer = self.create_timer(period, self._publish_voted_result)

        self.get_logger().info(
            f'\n\nHandGestureRecognition ready\n'
            f'  image topic   : {self.image_topic}\n'
            f'  model         : {self.model_path}\n'
            f'  voting window : {self.voting_window_sec}s\n'
            f'  publish rate  : {self.publish_rate_hz} Hz\n'
            f'  classes       : {list(self._label_encoder.classes_)}'
        )

    # Parameter declaration
    def _declare_params(self):
        self.declare_parameter('image_topic', '/head_front_camera/color/image_raw')
        self.declare_parameter('model_path', 'gesture_classifier.pkl')
        self.declare_parameter('landmarker_model', 'hand_landmarker.task')
        self.declare_parameter('min_confidence', 0.60)
        self.declare_parameter('num_hands', 2)
        self.declare_parameter('voting_window_sec', 0.75)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('detection_confidence', 0.50)
        self.declare_parameter('tracking_confidence', 0.50)
        self.declare_parameter('required_stability', 4)
        self.declare_parameter('show_preview', False)
        
    # Model loading
    def _load_classifier(self):
        # get package install/share directory
        package_path = get_package_share_directory('gesture_hri_pkg')

        # build correct absolute path
        path = os.path.join(package_path, 'models', self.model_path)
        path = os.path.abspath(path)

        self.get_logger().info(f'\n\nLoading classifier from: {path}\n')

        if not os.path.exists(path):
        	self.get_logger().fatal(f'Classifier not found: {path}')
        	raise FileNotFoundError(f'Missing model file: {path}')

        with open(path, 'rb') as f:
            self._classifier, self._label_encoder = pickle.load(f)

        self.get_logger().info(f'\n\nLoaded classifier classes={list(self._label_encoder.classes_)}\n')


    # MediaPipe initialisation
    def _init_landmarker(self):
        package_path = get_package_share_directory('gesture_hri_pkg')

        # build correct absolute path
        lm_path = os.path.join(package_path, 'models', self.landmarker_model)
        lm_path = os.path.abspath(lm_path)

        self.get_logger().info(f'\n\nLoading landmarker from: {lm_path}\n')
        
        if not os.path.exists(lm_path):
            self.get_logger().fatal(f'Hand landmarker model not found: {lm_path}')
            raise FileNotFoundError(f'Missing landmarker: {lm_path}')
            

        base_options = mp_python.BaseOptions(model_asset_path=lm_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.detect_conf,
            min_tracking_confidence=self.track_conf,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.get_logger().info('MediaPipe HandLandmarker initialised')

    # Image callback
    def _image_callback(self, msg: Image):
        """
        Convert ROS Image → MediaPipe Image → landmarks → classifier.
        Push each above-threshold prediction into the voting window.
        """
        try:
            frame = self._ros_image_to_numpy(msg)
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect(mp_img)
        except Exception as e:
            self.get_logger().warn(f'Landmark detection failed: {e}')
            return

        now = time.time()
        self._latest_hands = []
        
        preview = frame.copy() if self.show_preview else None

        if result.hand_landmarks:
            for hand_idx, hand_lms in enumerate(result.hand_landmarks):
                features, pixel_coords = normalize_landmarks(hand_lms, w, h)

                try:
                    proba = self._classifier.predict_proba([features])[0]
                except Exception as e:
                    self.get_logger().warn(f'Classifier error: {e}')
                    continue

                best_idx = int(np.argmax(proba))
                label    = str(self._label_encoder.classes_[best_idx])
                conf     = float(proba[best_idx])
                self.get_logger().debug(f"Prediction: {label} ({conf:.2f})")

                if label == self._last_label:
                    self._stable_count += 1
                else:
                    self._last_label = label
                    self._stable_count = 1

                # Always record per-hand result for the JSON "hands" field
                self._latest_hands.append({
                    'label':      label,
                    'confidence': round(conf, 4),
                    'hand_index': hand_idx,
                })

                # Only feed into voting window if above threshold and stable for enough frames
                if conf >= self.min_confidence and self._stable_count >= self.required_stability:
                    self._vote_window.append((label, conf, now))
                
                if self.show_preview and preview is not None:
                    # Draw landmark dots
                    for (px, py) in pixel_coords:
                        cv2.circle(preview, (px, py), 4, (0, 255, 0), -1)

                    # Draw connections between landmarks
                    for (a, b) in HAND_CONNECTIONS:
                        cv2.line(
                            preview,
                            pixel_coords[a],
                            pixel_coords[b],
                            (0, 200, 0), 1
                        )

                    # Label above the wrist (landmark 0)
                    wrist_x, wrist_y = pixel_coords[0]
                    text      = f"{label} ({conf:.2f})"
                    org       = (max(wrist_x - 30, 0), max(wrist_y - 15, 0))
                    fontscale = 1.0
                    thickness = 2
                    # Black outline for readability on any background
                    cv2.putText(preview, text, org,
                            cv2.FONT_HERSHEY_SIMPLEX, fontscale,
                            (0, 0, 0), thickness + 2)
                    cv2.putText(preview, text, org,
                            cv2.FONT_HERSHEY_SIMPLEX, fontscale,
                            (0, 255, 0), thickness)

                    # Stability bar — shows how many stable frames accumulated
                    bar_max   = self.required_stability
                    bar_val   = min(self._stable_count, bar_max)
                    bar_x     = max(wrist_x - 30, 0)
                    bar_y     = wrist_y + 10
                    bar_w     = 80
                    bar_h     = 8
                    filled    = int(bar_w * bar_val / bar_max)
                    cv2.rectangle(preview,
                            (bar_x, bar_y),
                            (bar_x + bar_w, bar_y + bar_h),
                            (50, 50, 50), -1)
                    cv2.rectangle(preview,
                            (bar_x, bar_y),
                            (bar_x + filled, bar_y + bar_h),
                            (0, 255, 100), -1)

        # Evict stale entries from the voting window
        cutoff = now - self.voting_window_sec
        while self._vote_window and self._vote_window[0][2] < cutoff:
            self._vote_window.popleft()
        
        if not result.hand_landmarks:
            self._last_label = None
            self._stable_count = 0
            
        if self.show_preview and preview is not None:
            # Show voted result at top of frame
            voted_label, voted_conf = majority_vote(self._vote_window)
            if voted_label is not None:
                voted_text = f"VOTED: {voted_label} ({voted_conf:.2f})"
                cv2.putText(preview, voted_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 0, 0), 4)
                cv2.putText(preview, voted_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (255, 220, 0), 2)

            num_hands_text = f"Hands: {len(self._latest_hands)}"
            cv2.putText(preview, num_hands_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 2)

            cv2.imshow("Gesture Recognition Preview", preview)
            # waitKey(1) is required to actually render the window — does not block
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Preview window closed by user")
                self.show_preview = False
                cv2.destroyAllWindows()


    # Publish
    def _publish_voted_result(self):
        """
        Compute majority vote over the current window and publish JSON.
        Skips publishing if the window is empty (no confident detections).
        """
        now = time.time()

        # Evict stale entries
        cutoff = now - self.voting_window_sec
        while self._vote_window and self._vote_window[0][2] < cutoff:
            self._vote_window.popleft()

        if not self._vote_window:
            # Nothing confident in the window – do not publish noise
            return

        voted_label, voted_conf = majority_vote(self._vote_window)

        if voted_label is None:
            return

        payload = {
            'label':      voted_label,
            'confidence': round(voted_conf, 4),
            'num_hands':  len(self._latest_hands),
            'hands':      self._latest_hands,
            'timestamp':  round(now, 4),
        }

        msg = String()
        msg.data = json.dumps(payload)
        self._pub.publish(msg)

        self.get_logger().debug(
            f'Published: label={voted_label}  conf={voted_conf:.2f}'
            f'  window_size={len(self._vote_window)}'
        )

    # Image format conversion
    def _ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        """
        Convert a sensor_msgs/Image to a BGR numpy array without cv_bridge.

        Supports encoding: rgb8, bgr8, mono8, 16UC1, 32FC1.
        Using manual conversion avoids the cv_bridge build dependency,
        which can be awkward in some ROS 2 / Python environments.
        """
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

    # Cleanup
    def destroy_node(self):
        self.get_logger().info('Shutting down HandGestureRecognition node')
        try:
            self._landmarker.close()
        except Exception:
            pass
        if self.show_preview:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HandGestureRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
