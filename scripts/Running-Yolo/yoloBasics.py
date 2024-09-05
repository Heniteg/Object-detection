from ultralytics import YOLO
import cv2

model = YOLO('../../Yolo-Weights/yolov8n.pt')
results = model("images/image3.png", show=True)
cv2.waitKey(0)