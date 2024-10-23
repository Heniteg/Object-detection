from ultralytics import YOLO
import cv2
import cvzone
import math

# Initialize the webcam
# cap = cv2.VideoCapture(0)

# Set the webcam resolution to 1280x720
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

# Open the video file for processing
cap = cv2.VideoCapture("../Videos/ppe-1.mp4")

# Load the pre-trained YOLO model (in this case, YOLOv8n)
model = YOLO("ppe.pt")

# classNames = ['boots', 'mask on', 'gloves', 'helmet', 'helmet on', 'no boots', 'no mask', 'no glove', 'no helmet', 'no vest', 'person', 'vest']
classNames = ['Hardhat', 'Mask', 'No-Hardhat', 'No-Mask', 'No-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

myColor = (0, 0, 255)

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
            # cvzone.cornerRect(img, (x1, y1, w, h), l=30, rt=5)

            # Using Opencv
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Adding the confidence 
            conf = math.ceil((box.conf[0] * 100)) / 100 # rouunding the confidence value to two digit 
            print(conf)
            # cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1))) # the max here is used to check the confidence level is not not out of the frame

            # Adding class name
            cls = int(box.cls[0]) # gives ID number of the class
            currentClass = classNames[cls] # gives the current class
            if conf > 0.5:  # to remove the unwanted class names
                if currentClass == 'No-Hardhat' or currentClass == 'No-Safety Vest' or currentClass == 'No-Mask':
                    myColor = (0, 0, 255) 
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == 'Mask':
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0) 

                cvzone.putTextRect(img, f'{classNames[cls]}: {conf}', (max(0, x1), max(35, y1)), 
                                scale=1.5, thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5) # the max here is used to check the confidence level is not not out of the frame

                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
           
    # Display the resulting frame
    cv2.imshow("Construction Site Safety Detection", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()