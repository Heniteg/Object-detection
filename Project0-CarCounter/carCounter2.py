# Import necessary libraries
from ultralytics import YOLO  # YOLO for object detection
import cv2  # OpenCV for image and video processing
import cvzone  # CvZone for enhanced OpenCV functionality
import numpy as np  # NumPy for numerical operations
import math  # Math for rounding operations
from sort import *  # SORT tracker for object tracking

# Open the video file for processing
cap = cv2.VideoCapture("../Videos/cars-demo.mp4")

# Load the pre-trained YOLO model for object detection
model = YOLO("../../Yolo-Weights/yolov8l.pt")

# Create a  line for counting the cars by setting the limits
limits = [350, 600, 1800, 600]

# Create a car counter
totalCount = [] # list

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
mask = cv2.imread("mask-2.png")
if mask is None:
    print("Error: Mask image not found.")
    exit()

# Initialize the SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
text_position = "below"  # Set the position for the text label relative to the bounding box

# Process each frame of the video
while True:
    success, img = cap.read()  # Capture the next video frame
    if not success:
        print("End of video or failed to capture image.")
        break

    # Adding image grraphics for the counting of car
    imgGraphics = cv2.imread("graphics-2.png", cv2.IMREAD_UNCHANGED)

    # Define the new size or scale
    new_width = 650 
    new_height = 200

    # Resize the image
    imgResized = cv2.resize(imgGraphics, (new_width, new_height), interpolation=cv2.INTER_AREA)

    img = cvzone.overlayPNG(img, imgResized, (0,0))

    # Resize the mask to match the frame's dimensions
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Ensure the mask has the same number of channels as the frame
    if len(mask_resized.shape) == 2:  # Convert grayscale to BGR if necessary
        mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the image
    imgRegion = cv2.bitwise_and(img, mask_resized)

    # Perform YOLO object detection on the masked image region
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))  # Initialize an array to hold detection data for tracking

    # Process each detected object
    for r in results:
        boxes = r.boxes  # Extract bounding boxes from the YOLO results
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert bounding box coordinates to integers
            w, h = x2 - x1, y2 - y1  # Calculate width and height of the bounding box

            # Convert confidence Tensor to float and round it
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])  # Retrieve the class ID for the detected object
            currentClass = classNames[cls]  # Map class ID to the class name

            # Filter for specific classes and confidence threshold
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                text = f'{currentClass}: {conf:.2f}'  # Create label with class and confidence

                # Determine text position based on the chosen setting
                if text_position == "above":
                    text_x, text_y = x1, y1 - 10
                    if text_y < 0:
                        text_y = y1 + 15
                elif text_position == "inside":
                    text_x, text_y = x1, y1 + 15
                elif text_position == "below":
                    text_x, text_y = x1, y2 + 20

                # Draw the text with background
                cvzone.putTextRect(img, text, (text_x, text_y), offset=5, scale=1, thickness=1)

                # Add the detection to the tracking array
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update the SORT tracker with the current frame's detections
    trackerResults = tracker.update(detections)

    # Draw the line 
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Display tracking results with bounding boxes and IDs
    for result in trackerResults:
        x1, y1, x2, y2, id = map(int, result)  # Unpack the tracked object's coordinates and ID
        w, h = x2 - x1, y2 - y1  # importnt to have a stable detection
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 0))  # Draw rectangle around the object
        cvzone.putTextRect(img, f'ID: {id}', (x1, y1), offset=5, scale=1, thickness=2)  # Display object ID

        # Finding the center of the detected object 
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED) # As soon as the circle creosses the line, we considered it to be detected

        # Fixing the counting problem
        if limits[0] <cx< limits[2] and limits[1] - 20 <cy< limits[1] + 20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 2255, 0), 5) # Green line color after beiing detected and counted
        
    # cvzone.putTextRect(img, f'Count: {int(len(totalCount))}', (50, 50), offset=10, scale=3, thickness=2)
    cv2.putText(img, str(len(totalCount)), (470,110), cv2.FONT_HERSHEY_PLAIN, 4, (60, 60, 255), 7)

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
