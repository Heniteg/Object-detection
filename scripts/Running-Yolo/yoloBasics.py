from ultralytics import YOLO
import cv2

model = YOLO('../../Yolo-Weights/yolov8l.pt')
results = model("images/image2.png", show=True)
cv2.waitKey(0)