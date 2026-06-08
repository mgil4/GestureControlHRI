#!/usr/bin/env python3
"""
Collect hand gesture data from a folder of static images
---------------------------------------------------------
Reads every image from the ASL dataset folder, runs MediaPipe Hand Landmarker
on each one, and writes the normalised landmarks to a CSV in the same format
as gesture_data.csv produced by collect_gestures.py.

Label is inferred from the filename: the leading letter(s)/digit(s) before
any trailing number, lowercased.
  Examples:
    A1.jpg      → a
    B.png       → b
    C12.jpg     → c
    0.jpg       → 0
    01.jpg      → 0

Output:
    extra_test_asl_data.csv

Run:
    python collect_from_images.py
"""

import os
import re
import csv
import time
import urllib.request

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image, ImageFormat

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

IMAGE_FOLDER   = r"C:\Users\maria\Downloads\dataset-asl-1\Combined"
OUTPUT_CSV     = "extra_test_asl_data.csv"

# How many times to "present" each image to the landmarker.
# MediaPipe is deterministic on a static image so 1 is enough for clean photos;
# raise to 3 if you want a quick retry on failed detections (same image, same
# result — but sometimes a second call succeeds on borderline cases).
DETECTION_RETRIES = 3

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LANDMARKER_MODEL = "hand_landmarker.task"
LANDMARKER_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def download_model(path, url):
    if not os.path.exists(path):
        print(f"[INFO] Downloading model → {path} ...")
        urllib.request.urlretrieve(url, path)
        print("[INFO] Done.")


def infer_label(filename):
    """
    Extract the gesture label from a filename.

    Rules:
      - Strip extension.
      - Take the leading non-digit characters if they exist → that is the label.
      - If the filename starts with a digit, take leading digits as label.
      - Lowercase the result.

    Examples:
      A1.jpg   → 'a'
      B.png    → 'b'
      C12.jpg  → 'c'
      0.jpg    → '0'
      01.jpg   → '0'
      10.jpg   → '1'   (first digit only, consistent with single-char labels)
    """
    stem = os.path.splitext(filename)[0]  # remove extension

    # Leading letters
    m = re.match(r'^([A-Za-z]+)', stem)
    if m:
        return m.group(1).lower()

    # Leading digits (single digit label)
    m = re.match(r'^(\d)', stem)
    if m:
        return m.group(1)

    # Fallback: use full stem lowercased
    return stem.lower()


def normalize_landmarks(landmarks, w, h):
    """
    Same normalization as collect_gestures.py:
      1. Convert to pixel coords.
      2. Make relative to wrist (landmark 0).
      3. Flatten to 1-D.
      4. Scale so max absolute value = 1.
    """
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    bx, by = coords[0]
    rel    = [(x - bx, y - by) for x, y in coords]
    flat   = [v for pair in rel for v in pair]
    mx     = max(abs(v) for v in flat) or 1
    return [v / mx for v in flat]


def build_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL)
    options      = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,   # slightly relaxed for photos
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_landmarks(landmarker, bgr_image):
    """
    Run detection on a BGR numpy image.
    Returns the first hand's landmark list, or None if nothing detected.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if result.hand_landmarks:
        return result.hand_landmarks[0]
    return None


def try_detect_with_preprocessing(landmarker, bgr_image):
    """
    Attempt detection with progressive preprocessing to help on tricky images.
    Returns landmarks or None.
    """
    h, w = bgr_image.shape[:2]

    attempts = [
        bgr_image,                                           # 1. original
        cv2.convertScaleAbs(bgr_image, alpha=1.3, beta=20), # 2. brighter / higher contrast
        cv2.resize(bgr_image, (max(w, 400), max(h, 400))),  # 3. upscaled (helps small hands)
    ]

    for img in attempts:
        for _ in range(DETECTION_RETRIES):
            lm = detect_landmarks(landmarker, img)
            if lm is not None:
                return lm
    return None


def collect_images_to_csv(image_folder, output_csv):
    download_model(LANDMARKER_MODEL, LANDMARKER_URL)

    # Gather all image files
    all_files = sorted([
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])

    if not all_files:
        print(f"[ERROR] No images found in {image_folder}")
        return

    print(f"[INFO] Found {len(all_files)} images in {image_folder}")

    landmarker = build_landmarker()

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
        writer.writerow(header)

        saved   = 0
        skipped = 0
        skip_log = {}   # label → count of skipped images

        for idx, filename in enumerate(all_files):
            label     = infer_label(filename)
            img_path  = os.path.join(image_folder, filename)
            bgr_image = cv2.imread(img_path)

            if bgr_image is None:
                print(f"  [WARN] Could not read image: {filename}")
                skipped += 1
                continue

            h, w = bgr_image.shape[:2]
            landmarks = try_detect_with_preprocessing(landmarker, bgr_image)

            if landmarks is None:
                skipped += 1
                skip_log[label] = skip_log.get(label, 0) + 1
                # Only log every 50 skips to avoid flooding the terminal
                if skipped <= 10 or skipped % 50 == 0:
                    print(f"  [SKIP] No hand detected: {filename}  (label={label})")
                continue

            features = normalize_landmarks(landmarks, w, h)
            writer.writerow([label] + features)
            csv_file.flush()
            saved += 1

            # Progress every 100 images
            if (idx + 1) % 100 == 0 or (idx + 1) == len(all_files):
                pct = (idx + 1) / len(all_files) * 100
                print(f"  [{idx+1:5d}/{len(all_files)}]  {pct:5.1f}%  "
                      f"saved={saved}  skipped={skipped}")

    landmarker.close()

    print(f"\n[DONE] Wrote {saved} rows to {output_csv}")
    print(f"[INFO] Skipped {skipped} images (no hand detected)")

    if skip_log:
        print("\nSkipped breakdown by label:")
        for lbl, cnt in sorted(skip_log.items()):
            print(f"  {lbl:6s}: {cnt} images skipped")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ASL IMAGE → LANDMARK CSV")
    print("=" * 55)
    print(f"  Source folder : {IMAGE_FOLDER}")
    print(f"  Output CSV    : {OUTPUT_CSV}")
    print()
    collect_images_to_csv(IMAGE_FOLDER, OUTPUT_CSV)


if __name__ == "__main__":
    main()
