from ultralytics import YOLO

def main():
    # 1. Load your freshly trained weights
    # Make sure this path points to your new train-4 folder
    model = YOLO(r"runs\detect\train-4\weights\best.pt")

    # 2. Run inference on a static screenshot
    # Replace 'test_image.jpg' with the path to a screenshot
    results = model.predict(
        source=r"C:\Users\yashm\vla_highlight_agent\datasets\valid\images\A1671468090503-532959_jpg.rf.5800be34786c342a228f97551c6935d3.jpg",
        conf=0.5,      # Only show detections with 50%+ confidence
        save=True,     # Saves the output image with bounding boxes drawn
        show=True      # Pops open a window to show you the result
    )

    print("Testing complete. Check the 'runs/detect/predict' folder for the saved image.")

if __name__ == '__main__':
    main()