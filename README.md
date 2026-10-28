# Valorant Highlight Agent

An agent that watches your Valorant gameplay in real time and automatically saves highlight clips when you get a kill.

## How it works
- **Capture:** grabs the screen at 1080p with `mss` and keeps a rolling buffer of recent frames.
- **Detect:** a YOLOv8 model, plus a colour check on the kill-banner region, spots kill moments.
- **Clip:** saves the moments before and after each kill to MP4 in a background thread, so capture never lags.

## Training
- `extract_frames.py`: pulls 1 frame per second from your own gameplay videos
- `train_yolo.py`: trains YOLOv8 on a 5,200+ image Valorant dataset ([Roboflow, CC BY 4.0](https://universe.roboflow.com/fps/valorant-gygiq/dataset/6))
- `test_static.py`: checks the model on still images

## Tech stack
Python · Ultralytics YOLOv8 · OpenCV · NumPy · mss

## Run
```bash
pip install ultralytics opencv-python mss numpy
python train_yolo.py     # train the model
python live_agent.py     # start clipping (Windows, 1920x1080)
```
