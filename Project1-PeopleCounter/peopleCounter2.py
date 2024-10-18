# Import necessary libraries
from ultralytics import YOLO  # YOLO for object detection
import cv2  # OpenCV for image and video processing
import cvzone  # CvZone for enhanced OpenCV functionality
import numpy as np  # NumPy for numerical operations
import math  # Math for rounding operations
from sort import *  # SORT tracker for object tracking

# Open the video file for processing
cap = cv2.VideoCapture("../Videos/people.mp4")

# Load the pre-trained YOLO model for object detection
model = YOLO("../../Yolo-Weights/yolov8l.pt")

# Create a  line for counting the cars by setting the limits
limitsUp = [103, 161, 296, 161] # people moving upward
limitsDown = [527, 489, 735, 489] # people going downward

# Create a car counter
totalCountUp = [] # list
totalCountDown = []

# Define the list of object class names YOLO can detect
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load the mask image to define the region of interest
mask = cv2.imread("mask-people.jpg")
if mask is None:
    print("Error: Mask image not found.")
    exit()

# Initialize the SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
text_position = "below"  # Set the position for the text label relative to the bounding box

# Process each frame of the video
while True:
    success, img = cap.read()  # Capture the next video frame
    # Apply the mask to the image
    imgRegion = cv2.bitwise_and(img, mask)

    if not success:
        print("End of video or failed to capture image.")
        break


    # Adding image grraphics for the counting of car
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730,260))
    
    # Perform YOLO object detection on the masked image region
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))  # Initialize an array to hold detection data for tracking

    # Process each detected object
    for r in results:
        boxes = r.boxes  # Extract bounding boxes from the YOLO results
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert bounding box coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1  # Calculate width and height of the bounding box

            # Convert confidence Tensor to float and round it
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])  # Retrieve the class ID for the detected object
            currentClass = classNames[cls]  # Map class ID to the class name

            # Filter for specific classes and confidence threshold
            if currentClass in ["person"] and conf > 0.3:
                text = f'{currentClass}: {conf:.2f}'  # Create label with class and confidence

                # Determine text position based on the chosen setting
                if text_position == "above":
                    text_x, text_y = x1, y1 - 10
                    if text_y < 0:
                        text_y = y1 + 15
                elif text_position == "inside":
                    text_x, text_y = x1, y1 + 15
                elif text_position == "below":
                    text_x, text_y = x1, y2 + 15

                # Draw the text with background
                cvzone.putTextRect(img, text, (text_x, text_y), offset=3, scale=1, thickness=1)

                # Add the detection to the tracking array
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                
    # Update the SORT tracker with the current frame's detections
    trackerResults = tracker.update(detections)

    # Draw the line 
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    # Display tracking results with bounding boxes and IDs
    for result in trackerResults:
        x1, y1, x2, y2, id = map(int, result)  # Unpack the tracked object's coordinates and ID
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))  # Draw rectangle around the object
        cvzone.putTextRect(img, f'ID: {id}', (max(0, x1), max(35, y1)), offset=6, scale=2, thickness=2)  # Display object ID

        # Finding the center of the detected object 
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED) # As soon as the circle creosses the line, we considered it to be detected

        # Fixing the counting problem
        # For people mving upwards
        if limitsUp[0] <cx< limitsUp[2] and limitsUp[1] - 20 <cy< limitsUp[1] + 20:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 2255, 0), 5) # Green line color after being detected and counted
        
        # For people mving downwards
        if limitsDown[0] <cx< limitsDown[2] and limitsDown[1] - 20 <cy< limitsDown[1] + 20:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 2255, 0), 5) # Green line color after being detected and counted
        
    # cvzone.putTextRect(img, f'Count: {int(len(totalCount))}', (50, 50), offset=10, scale=3, thickness=2)
    cv2.putText(img, str(len(totalCountUp)), (929,345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 185, 75), 7) # for moving up
    cv2.putText(img, str(len(totalCountDown)), (1191,345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7) # for moving down

    # Display the frames with detected objects and masked region
    cv2.imshow("YOLO Object Detection", img)
    # cv2.imshow("ImageRegionOfInterest", imgRegion)

    # cv2.waitKey(0) # Wait for the keyboard

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
