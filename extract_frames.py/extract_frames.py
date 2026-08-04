import cv2
import os
import glob

# Ensure output directory exists for the extracted images
os.makedirs("dataset_frames", exist_ok=True)

# Look for mp4 files in the current folder AND in raw_clips/
video_files = glob.glob("*.mp4") + glob.glob("raw_clips/*.mp4")

print(f"Found {len(video_files)} video(s): {video_files}")
print("Starting extraction...")

frame_count = 0

for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    
    # Get FPS safely
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print(f"Skipping {video_path} (could not read video/FPS)")
        continue
    fps = round(fps)
    
    success, frame = cap.read()
    current_frame = 0
    
    while success:
        # Extract exactly 1 frame per second
        if current_frame % fps == 0:
            cv2.imwrite(f"dataset_frames/frame_{frame_count}.jpg", frame)
            frame_count += 1
            
        success, frame = cap.read()
        current_frame += 1
        
    cap.release()

print(f"\nExtraction complete! {frame_count} unique frames saved to 'dataset_frames'.")