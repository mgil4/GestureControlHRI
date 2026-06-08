# Gesture HRI Package

ROS 2 gesture-controlled human-robot interaction system for TIAGo Pro.
Three interaction modes controlled by a single launch parameter.

## Dependencies

ROS 2 Humble. Python dependencies:
```bash
pip install -r src/gesture_hri_pkg/requirements.txt --break-system-packages
```

For LLM mode, install Ollama and pull the model:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:0.6b
```

## Model files

Place these in `src/gesture_hri_pkg/models/` before building:
- `gesture_classifier.pkl`
- `hand_landmarker.task`

`hand_landmarker.task` can be downloaded from the MediaPipe releases page.
`gesture_classifier.pkl` is the trained ASL classifier (distribute separately).

## Build

```bash
cd ~/gesture_hri_project
colcon build --packages-select gesture_hri_pkg
source install/setup.bash
```

## Run

**Mode 1 — Robot command:**
```bash
ros2 launch gesture_hri_pkg gesture_system.launch.py mode:=robot
```

**Mode 2A — Sign to speech:**
```bash
ros2 launch gesture_hri_pkg gesture_system.launch.py mode:=text
```

**Mode 2B — Sign to conversation:**
```bash
ollama serve  # separate terminal
ros2 launch gesture_hri_pkg gesture_system.launch.py mode:=llm
```

**On the real robot**, override the image topic:
```bash
ros2 launch gesture_hri_pkg gesture_system.launch.py \
  mode:=robot \
  image_topic:=/head_front_camera/color/image_raw
```
