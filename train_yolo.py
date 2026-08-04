from ultralytics import YOLO

def main():
    # Load the base YOLO11 nano model
    model = YOLO("yolo11n.pt")
    
    # Start a fresh 30-epoch run on the GPU
    results = model.train(
        data="datasets/data.yaml", 
        epochs=30, 
        device=0
    )

if __name__ == '__main__':
    main()