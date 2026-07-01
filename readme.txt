1. Train (default 30 epochs):
python main.py train
Train with custom epochs:
python main.py train --epochs 50


2. Recognize (auto-picks latest weights):
python main.py recognize
Recognize with specific weights:
python main.py recognize --weights runs/detect/train-11/weights/best.pt
