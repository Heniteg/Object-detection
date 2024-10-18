# Car and People Counting Project Using YOLOv8 with GPU

This project demonstrates a car and people counting application using YOLOv8 for object detection, SORT for object tracking, and a GPU for accelerated processing. The system detects and counts vehicles (cars, buses, trucks, and motorbikes) and people within a specific region, defined by custom masks designed in Canva.

## Project Overview

The primary goal of this project is to count both vehicles and people within designated areas in real-time. By leveraging YOLOv8, we can achieve accurate and efficient object detection. The SORT algorithm is used to track the detected objects over frames to ensure accurate counting, with unique IDs assigned to each object.

### Key Features
- **Vehicle Detection**: Utilizes YOLOv8 for detecting cars, buses, trucks, and motorbikes.
- **People Detection**: Detects people and counts them as they move up or down within a defined region.
- **Custom Masking**: Uses Canva to design region masks where counting takes place for both vehicles and people.
- **Object Tracking**: Implements the SORT algorithm from the GitHub repository to track objects (vehicles and people) frame-by-frame with unique IDs.
- **Vehicle and People Counting**: Counts the number of vehicles and people passing through the defined regions in real-time.
- **Real-time Processing**: Runs on a GPU for high-performance and real-time processing.

## Workflow

1. **Mask Design**: 
   - Two regions of interest are defined using masks created in Canva: one for vehicle counting and one for people counting. These masks allow the application to focus on specific areas for counting.
   
2. **Detection**:
   - YOLOv8 is used to detect vehicles and people in each frame. It identifies the vehicle type (car, bus, truck, motorbike) and tracks people moving up or down.
   
3. **Tracking**:
   - The SORT algorithm tracks both detected vehicles and people across frames, ensuring consistency in counting and minimizing duplicate counts. Each object (vehicle or person) is assigned a unique ID.
   
4. **Counting**:
   - Vehicles are counted as they enter and exit the masked region. Similarly, people are counted as they move up or down within the defined region. The system keeps a tally of each vehicle type and the number of people moving in each direction.

5. **OpenCV**:
   - OpenCV handles video input/output, processes frames, and manages the real-time display of results for both vehicles and people.

## Setup Instructions

### Prerequisites
- Python 3.x
- GPU with CUDA support
- [ultralytics](https://github.com/ultralytics/)
- [SORT GitHub Repository](https://github.com/abewley/sort)
- [OpenCV](https://opencv.org/)
- Canva (for mask creation)

### Sample Outputs

#### **YOLOv8l (large) car counting Output**

![YOLOv8l Car_Counting_Output](https://github.com/Heniteg/Object-detection/blob/counter-with-yolo/Project0-CarCounter/output/car-counter.gif)

#### **YOLOv8l (large) people counting Output**

![YOLOv8l People_Counting_Output](https://github.com/Heniteg/Object-detection/blob/counter-with-yolo/Project0-CarCounter/output/people-counter.gif)

#### **Masking for the region of interest (Car Counting)**

![Masking](https://github.com/Heniteg/Object-detection/blob/counter-with-yolo/Project0-CarCounter/output/mask-2.png)

#### **Masking for the region of interest (People Counting)**

![Masking](https://github.com/Heniteg/Object-detection/blob/counter-with-yolo/Project0-CarCounter/output/mask-people.jpg)

## How to Use

To replicate the experiment:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Heniteg/Object-detection.git
    cd Object-detection
    git checkout counter-with-yolo
    ```

2. **Run the object detection on video**:
    - Modify the model weights and input video path inside `video_detection.py`:
        ```python
        model = YOLO("weights/yolov8l.pt")  # You can switch to any other model
        video_path = "path_to_your_video/video.mp4"
        ```

    - Then run the script:
        ```bash
        python Project0-CarCounter/carCounter2.py
        ```

3. **For People Counting**:
    - Similar to vehicle counting, modify the model weights and input video path inside the corresponding script for people counting:
        ```python
        model = YOLO("weights/yolov8l.pt")  # You can switch to any other model
        video_path = "path_to_your_people_video/video.mp4"
        ```

    - Then run the script:
        ```bash
        python Project1-PeopleCounter/peopleCounter.py
        ```

---

This will allow the system to count vehicles and people, tracking their movements with unique IDs and using real-time processing for both applications.
