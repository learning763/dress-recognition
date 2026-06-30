from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    print("--- 🚀 Training the AI... ---")
    results = model.train(
        data="dataset.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        single_cls=True,
        conf=0.1,
        iou=0.3,
        augment=True,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=10,
        translate=0.15,
        scale=0.6,
        shear=2,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        lr0=0.005,
        lrf=0.001,
        patience=100,
        close_mosaic=0,
        box=7.5,
        cls=0.7,
        dfl=1.5,
        seed=42,
        deterministic=True,
    )

    save_dir = results.save_dir
    best_weights = f"{save_dir}/weights/best.pt"
    print(f"\n--- 🎉 Training Complete! Best weights: {best_weights} ---")

    custom_brain = YOLO(best_weights)

    print("\n--- 👀 Running Recognition on test.jpeg... ---")
    pred = custom_brain.predict(
        source="test.jpeg",
        save=True,
        show=True,
        conf=0.05,
        iou=0.2,
        max_det=50,
        augment=True,
        agnostic_nms=False,
    )
    
    boxes = pred[0].boxes
    print(f"\nDetections found: {len(boxes)}")
    if len(boxes) == 0:
        print("WARNING: No objects detected. Possible fixes:")
        print("  1. Ensure test.jpeg is one of your labeled training images")
        print("  2. Add more training images (aim for 100+)")
        print("  3. Check that labels actually enclose the object in each image")
        print("  4. Try increasing epochs to 200+")
    else:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls)
            cls_name = custom_brain.names[cls_id]
            conf = box.conf.item()
            xyxy = box.xyxy[0].tolist()
            print(f"  [{i+1}] {cls_name}: confidence={conf:.3f}, box={[round(v,1) for v in xyxy]}")

    print(f"\nOutput saved to: {pred[0].save_dir}")

if __name__ == "__main__":
    main()
