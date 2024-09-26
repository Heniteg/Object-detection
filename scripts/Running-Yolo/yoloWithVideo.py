from ultralytics import YOLO
import cv2
import cvzone
import math

# Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Set the webcam resolution to 1280x720
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

cap = cv2.VideoCapture("../../Videos/cars.mp4")

# Load the pre-trained YOLO model (in this case, YOLOv8n)
model = YOLO("../../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    # Capture the frame from the webcam
    success, img = cap.read()

    # Check if frame was captured successfully
    if not success:
        print("Failed to capture image.")
        break

    # Run the YOLO model on the captured frame and stream results
    results = model(img, stream=True)

    # Iterate over the detected objects in the results
    for r in results:
        boxes = r.boxes  # Access the bounding boxes for detected objects

        # Loop through all the detected bounding boxes
        for box in boxes:
            # Get coordinates of the bounding box (top-left and bottom-right corners)
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers

            # Calculate the width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Draw a rectangle around the detected object using cvzone
            cvzone.cornerRect(img, (x1, y1, w, h), l=30, rt=5)

            # Using Opencv
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Adding the confidence 
            conf = math.ceil((box.conf[0] * 100)) / 100 # rouunding the confidence value to two digit 
            print(conf)
            # cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1))) # the max here is used to check the confidence level is not not out of the frame

            # Adding class name
            cls = int(box.cls[0]) # gives ID number of the class

            cvzone.putTextRect(img, f'{classNames[cls]}: {conf}', (max(0, x1), max(35, y1))) # the max here is used to check the confidence level is not not out of the frame


    # Display the resulting frame
    cv2.imshow("YOLO Object Detection", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()