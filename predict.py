from ultralytics import YOLO
import numpy as np
import os

model = YOLO(r'runs\classify\train2\weights\best2.pt')

# def classify_image(image_file):
results = model(r"C:\Users\athik\OneDrive\Desktop\Study Material\AI Project\785.JPEG")
names_dict = results[0].names
probs = results[0].probs.data.tolist()

print(names_dict)
print(probs)
print(names_dict[np.argmax(probs)])
class_name = names_dict[np.argmax(probs)]
print(class_name)
# return class_name
