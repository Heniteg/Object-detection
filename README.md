# Car, People Counting, and PPE Detection Projects Using YOLOv8 with GPU

This repository demonstrates three projects utilizing YOLOv8 for object detection, SORT for object tracking, OpenCV for video processing, and a GPU for accelerated performance:
1. Car counting
2. People counting
3. PPE (Personal Protective Equipment) detection

## Project Overview

The primary goals of these projects are to count vehicles and people within designated areas in real-time, as well as detect the presence of construction safety equipment (PPE) such as helmets, vests, gloves, and boots in a work environment. By leveraging YOLOv8, we can achieve accurate and efficient object detection. The SORT algorithm is used to track detected objects over frames to ensure accurate counting and detection with unique IDs assigned to each object.

### Key Features
- **Vehicle Detection**: Utilizes YOLOv8 for detecting cars, buses, trucks, and motorbikes.
- **People Detection**: Detects people and counts them as they move up or down within a defined region.
- **PPE Detection**: Detects construction safety equipment like helmets, vests, gloves, and boots.
- **Custom Masking**: Uses Canva to design region masks where counting and detection take place.
- **Object Tracking**: Implements the SORT algorithm to track objects frame-by-frame with unique IDs.
- **Real-time Processing**: Runs on a GPU for high-performance and real-time processing.
  
## PPE (Personal Protective Equipment) Detection

This project detects construction safety equipment using a dataset provided by [Roboflow](https://roboflow.com/). The goal is to ensure compliance with safety regulations by identifying whether workers are wearing the appropriate PPE such as helmets, vests, gloves, and mask.

### Dataset
- The dataset used for training the model is provided by Roboflow, which includes labeled images of workers wearing PPE in various construction environments.

### Workflow
1. **Data Preparation**:
   - The dataset from Roboflow is used to train YOLOv8 for detecting PPE like helmets, gloves, vests, and boots.
   
2. **Detection**:
   - YOLOv8 is used to detect the presence of PPE in each frame of a video. It identifies the type of PPE being worn and ensures compliance with safety standards.
   
3. **Result**:
   - The system successfully detects PPE on workers in a real-time or recorded video. Each detected object (PPE item) is tracked and marked to ensure proper safety equipment is used.

---

## Car and People Counting Workflow

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
- [Roboflow](https://roboflow.com/) dataset for PPE detection

### Sample Outputs

#### **YOLOv8l (large) car counting Output**

![YOLOv8l Car_Counting_Output](https://github.com/Heniteg/Object-detection/blob/counter-with-yolo/Project0-CarCounter/output/car-counter.gif)

#### **YOLOv8l (large) people counting Output**

![YOLOv8l People_Counting_Output](https://github.com/Heniteg/Object-detection/blob/counter-with-yolo/Project1-PeopleCounter/output/people-counter.gif)

#### **PPE Detection Output**

![PPE Detection Output](https://github.com/Heniteg/Object-detection/blob/counter-with-yolo/Project2-PPEDetection/output/ppe-detector.gif)

#### **Masking for the region of interest (Car Counting)**

![Masking](https://github.com/Heniteg/Object-detection/blob/counter-with-yolo/Project0-CarCounter/output/mask-2.png)

#### **Masking for the region of interest (People Counting)**

![Masking](https://github.com/Heniteg/Object-detection/blob/counter-with-yolo/Project1-PeopleCounter/output/people-mask.jpg)

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
        python Project0-CarCounter/peopleCounter.py
        ```

4. **For PPE Detection**:
    - Modify the model weights and input video path inside the corresponding script for PPE detection:
        ```python
        model = YOLO("ppe.pt")  # You can switch to any other model
        video_path = "path_to_your_ppe_video/video.mp4"
        ```

    - Then run the script:
        ```bash
        python Project0-PPEDetector/ppeDetector.py
        ```

---

The above updated README includes car counting, people counting, and PPE detection functionalities using YOLOv8, SORT, and OpenCV, with detailed instructions on how to set up and run each project. The PPE detection project uses the Roboflow dataset for training the model to detect construction safety equipment.
