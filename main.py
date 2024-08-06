from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(data=r'C:\Users\athik\OneDrive\Desktop\Study Material\AI Project\dataset',epochs=20,imgsz=64)