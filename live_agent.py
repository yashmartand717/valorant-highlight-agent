import cv2
import numpy as np
import mss
import time
import os
import ctypes
import threading
from collections import deque
from ultralytics import YOLO

# Force Windows to give the raw, unscaled screen resolution
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

def save_clip_background(filename, frames_to_save, fps):
    """Saves the video in a background thread so the main game capture doesn't lag."""
    print(f"\n[Background Thread] Encoding video at {fps:.2f} FPS... Please don't close the script.")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(filename, fourcc, fps, (1920, 1080))
    
    for frame in frames_to_save:
        out_writer.write(frame)
        
    out_writer.release()
    print(f"\n[Background Thread] Successfully saved {filename}!")

def check_kill_banner(frame):
    if frame.shape[0] < 1080 or frame.shape[1] < 1920:
        return False
        
    roi = frame[850:930, 920:1000]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([0, 150, 150])
    upper_bound = np.array([179, 255, 255])
    
    mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
    matching_pixels = cv2.countNonZero(mask)
    
    return matching_pixels > 300

def main():
    model = YOLO(r"runs\detect\train-4\weights\best.pt")
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    
    # --- TIME CROPPING CONFIGURATION ---
    # Edit these to change how much time is recorded around a kill!
    pre_kill_seconds = 1.0   # Records exactly 1 second before the kill
    post_kill_seconds = 1.5  # Records exactly 1.5 seconds after the kill
    # -----------------------------------
    
    # Store about 60 seconds of memory just to be safe
    frame_buffer = deque(maxlen=2000) 
    
    recording = False
    frames_left_to_record = 0
    kill_cooldown = 0 
    clip_frames = [] 
    
    os.makedirs("highlights", exist_ok=True)
    print("Starting VLA Highlight Agent (Anti-Lag Threading). Press 'Ctrl+C' in terminal to quit.")

    fps_start_time = time.time()
    fps_counter = 0
    actual_fps = 30.0 

    with mss.mss() as sct:
        while True:
            # Calculate dynamic FPS
            fps_counter += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                actual_fps = max(15.0, fps_counter / elapsed_time) 
                fps_counter = 0
                fps_start_time = time.time()

            if kill_cooldown > 0:
                kill_cooldown -= 1

            # Grab screen
            screenshot = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            # Predict enemies (Boxes are evaluated but purposely NOT drawn on screen to save heavy CPU overhead)
            results = model.predict(frame, imgsz=640, half=True, verbose=False, stream=True)
            annotated_frame = frame.copy()
            
            # Loop runs to clear generator, but r.plot() is removed to drastically increase FPS
            for r in results:
                pass 

            frame_buffer.append(annotated_frame)
            kill_secured = check_kill_banner(frame)

            # --- Trigger Logic ---
            if kill_secured and kill_cooldown <= 0:
                kill_cooldown = int(actual_fps * 0.5) 
                
                if not recording:
                    print(f"Kill Triggered! Capturing frames...")
                    recording = True
                    
                    # Grab the past 'X' seconds from memory
                    frames_to_dump = int(actual_fps * pre_kill_seconds)
                    recent_frames = list(frame_buffer)[-frames_to_dump:] 
                    clip_frames.extend(recent_frames)
                else:
                    print("Multi-kill! Extending the clip duration...")
                
                # Set post-kill timer
                frames_left_to_record = int(actual_fps * post_kill_seconds)
            
            # --- Continuous Recording Phase ---
            if recording:
                clip_frames.append(annotated_frame)
                frames_left_to_record -= 1
                
                if frames_left_to_record <= 0:
                    recording = False
                    filename = f"highlights/clip_{int(time.time())}.mp4"
                    
                    # SPAWN BACKGROUND THREAD TO SAVE VIDEO
                    # This completely stops the game from lagging when a clip saves
                    thread = threading.Thread(target=save_clip_background, args=(filename, clip_frames, actual_fps))
                    thread.start()
                    
                    clip_frames = [] 

            # NOTE: cv2.imshow() is completely removed! 
            # Drawing a video player on your screen costs a massive amount of FPS.
            # The script now runs invisibly and powerfully in the background terminal.

if __name__ == "__main__":
    main()