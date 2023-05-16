from ultralytics import YOLO
import cv2
model=YOLO("best.pt")

img=cv2.imread("1067.jpg")

sonuc = model.predict(source=img,save=True,conf=0.5,iou=0.5)
print(sonuc)

