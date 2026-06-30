from ultralytics import YOLO

def main():
    # 1. Start with a baseline smart model template
    model = YOLO("yolov8n.pt")

    # 2. Teach the AI using your Makesense folder data
    print("--- 🚀 Step A: Training the AI... ---")
    model.train(data="dataset.yaml", epochs=1000, imgsz=640)
    print("--- 🎉 Step B: Training Complete! ---")

    # 3. Load your brand new custom trained weights
    # (YOLO automatically creates and saves this file during training)
    custom_brain = YOLO("runs/detect/train/weights/best.pt")

    # 4. Feed a completely new, unlabeled photo to recognize objects!
    print("--- 👀 Step C: Running Recognition... ---")
    results = custom_brain.predict(source="armytest.webp", save=True, show=True)
    
    print("All done! Look in the 'runs/detect/predict/' folder to see the boxed image.")

if __name__ == "__main__":
    main()


    