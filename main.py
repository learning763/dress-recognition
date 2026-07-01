import argparse
import glob
import os
import cv2
from ultralytics import YOLO


DISPLAY_W, DISPLAY_H = 800, 450
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.4
MAX_BOX_RATIO = 0.90
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 255, 255)}
# green=Combat, blue=Official, red=Civil, yellow=Tunik


def _suppress_contained(boxes, contain_thresh=0.8):
    if len(boxes) == 0:
        return []

    coords = [
        list(map(int, b.xyxy[0].tolist())) + [b.conf.item(), int(b.cls.item())]
        for b in boxes
    ]
    coords.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    kept = []
    for b in coords:
        x1, y1, x2, y2, conf, cls_id = b
        b_area = (x2 - x1) * (y2 - y1)
        dominated = False
        for kx1, ky1, kx2, ky2, *_ in kept:
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if b_area > 0 and inter / b_area >= contain_thresh:
                dominated = True
                break
        if not dominated:
            kept.append((x1, y1, x2, y2, conf, cls_id))
    return kept


def _latest_weights():
    candidates = glob.glob("runs/detect/train*/weights/best.pt")
    if not candidates:
        raise FileNotFoundError("No trained weights found. Run in train mode first.")
    latest = max(candidates, key=os.path.getmtime)
    print(f"Auto-selected weights: {latest}")
    return latest


def train(epochs=30):
    # delete stale label caches so YOLO re-reads all label files
    for cache in glob.glob("dataset_files/**/labels.cache", recursive=True):
        os.remove(cache)
        print(f"Removed stale cache: {cache}")

    print(f"--- Training for {epochs} epochs ---")
    model = YOLO("yolov8s.pt")
    results = model.train(
        data="dataset.yaml",
        epochs=epochs,
        imgsz=640,
        batch=8,
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

    best_weights = f"{results.save_dir}/weights/best.pt"
    print(f"--- Training complete. Weights saved to: {best_weights} ---")


def recognize(weights=None):
    if weights is None:
        weights = _latest_weights()
    print(f"Loading weights: {weights}")
    model = YOLO(weights)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    cv2.namedWindow("YOLOv8 Real-Time Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Real-Time Detection", DISPLAY_W, DISPLAY_H)
    print("Camera open — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        results = model.predict(frame, conf=0.25, iou=IOU_THRESHOLD, verbose=False)
        raw = results[0].boxes
        if len(raw):
            for b in raw:
                print(f"  raw: {model.names[int(b.cls)]} conf={b.conf.item():.2f}")
        kept = _suppress_contained([b for b in raw if b.conf.item() >= CONF_THRESHOLD])

        fh, fw = frame.shape[:2]
        frame_area = fw * fh
        annotated = frame.copy()

        for x1, y1, x2, y2, conf, cls_id in kept:
            if (x2 - x1) * (y2 - y1) / frame_area > MAX_BOX_RATIO:
                continue
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("YOLOv8 Real-Time Detection", cv2.resize(annotated, (DISPLAY_W, DISPLAY_H)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "recognize"],
                        help="train: train the model | recognize: run live detection")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (only used in train mode)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file (only used in recognize mode)")
    args = parser.parse_args()

    if args.mode == "train":
        train(epochs=args.epochs)
    else:
        recognize(weights=args.weights)
  