from ultralytics import YOLO

model = YOLO('Weight/best.pt')

results = model(source=0, show=True, conf=0.6, save=False)