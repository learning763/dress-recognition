"""
Run this script to visually assign the correct class to each image.

Controls:
  0  →  mark as Combat    (class 0, green label)
  1  →  mark as Official  (class 1, blue label)
  2  →  mark as Civil     (class 2, red label)
  3  →  mark as Tunik     (class 3, yellow label)
  s  →  skip this image (leave label unchanged)
  q  →  quit and save progress

Usage:
  python fix_labels.py
"""

import os
import glob
import cv2

IMAGE_DIRS = [
    "dataset_files/train/images",
    "dataset_files/val/images",
]
CLASS_NAMES = {0: "Combat", 1: "Official", 2: "Civil", 3: "Tunik"}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 255, 255)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def label_path_for(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    label_dir = img_path.replace("images", "labels", 1)
    label_dir = os.path.dirname(label_dir)
    return os.path.join(label_dir, stem + ".txt")


def read_label(lp):
    if not os.path.exists(lp):
        return []
    with open(lp) as f:
        return [line.strip() for line in f if line.strip()]


def write_label(lp, lines):
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    with open(lp, "w") as f:
        f.write("\n".join(lines) + "\n")


def change_class(lines, new_cls):
    result = []
    for line in lines:
        parts = line.split()
        if parts:
            parts[0] = str(new_cls)
        result.append(" ".join(parts))
    return result


def make_full_box_line(cls_id):
    return f"{cls_id} 0.5 0.5 1.0 1.0"


def collect_images():
    images = []
    for d in IMAGE_DIRS:
        for ext in IMG_EXTS:
            images += glob.glob(os.path.join(d, f"*{ext}"))
            images += glob.glob(os.path.join(d, f"*{ext.upper()}"))
    return sorted(set(images))


def current_class(lines):
    for line in lines:
        parts = line.split()
        if parts:
            try:
                return int(parts[0])
            except ValueError:
                pass
    return None


def main():
    images = collect_images()
    print(f"Found {len(images)} images total.")

    cv2.namedWindow("Label Fixer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Label Fixer", 900, 600)

    i = 0
    while i < len(images):
        img_path = images[i]
        lp = label_path_for(img_path)
        lines = read_label(lp)
        cls = current_class(lines)

        img = cv2.imread(img_path)
        if img is None:
            i += 1
            continue

        display = img.copy()
        h, w = display.shape[:2]

        # show current class banner
        cls_name = CLASS_NAMES.get(cls, "NO LABEL")
        color = CLASS_COLORS.get(cls, (128, 128, 128))
        cv2.rectangle(display, (0, 0), (w, 50), color, -1)
        cv2.putText(display, f"[{i+1}/{len(images)}] {os.path.basename(img_path)}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(display, f"Current: {cls_name}  |  Press 0=Combat 1=Official 2=Civil 3=Tunik  s=skip  q=quit",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("Label Fixer", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("Quit. Progress saved.")
            break
        elif key in (ord('0'), ord('1'), ord('2'), ord('3')):
            new_cls = key - ord('0')
            new_lines = change_class(lines, new_cls) if lines else [make_full_box_line(new_cls)]
            write_label(lp, new_lines)
            print(f"  {CLASS_NAMES[new_cls]} → {os.path.basename(img_path)}")
            i += 1
        elif key == ord('s'):
            print(f"  skip   → {os.path.basename(img_path)}")
            i += 1

    cv2.destroyAllWindows()
    print("\nDone. Now retrain with: python main.py train --epochs 50")


if __name__ == "__main__":
    main()
