# Car Counting Project Using YOLOv8 with GPU

This project demonstrates a car counting application using YOLOv8 for object detection, SORT for object tracking, and a GPU for accelerated processing. The system detects and counts vehicles (cars, buses, trucks, and motorbikes) within a specific region, defined by a custom mask designed in Canva.

## Project Overview

The primary goal of this project is to count vehicles within a designated area in real-time. By leveraging YOLOv8, we can achieve accurate and efficient object detection. The SORT algorithm is then used to track the detected objects over frames to ensure accurate counting.

### Key Features
- **Vehicle Detection**: Utilizes YOLOv8 for detecting cars, buses, trucks, and motorbikes.
- **Custom Masking**: Uses Canva to design a region mask where counting takes place.
- **Object Tracking**: Implements the SORT algorithm from the GitHub repository to track objects frame-by-frame.
- **Vehicle Counting**: Counts the number of vehicles passing through the defined region in real-time.
- **Real-time Processing**: Runs on a GPU for high-performance and real-time processing.

## Workflow

1. **Mask Design**: 
   - The region of interest is defined using a mask created in Canva. This mask allows the application to focus on a specific area for vehicle counting.
   
2. **Detection**:
   - YOLOv8 is used to detect vehicles in each frame. It identifies the vehicle type (car, bus, truck, motorbike) and provides bounding boxes for each detected object.
   
3. **Tracking**:
   - The SORT algorithm tracks the detected vehicles across frames, ensuring consistency in counting and minimizing duplicate counts.
   
4. **Counting**:
   - Vehicles are counted as they enter and exit the masked region. The system keeps a tally of each vehicle type.

## Setup Instructions

### Prerequisites
- Python 3.x
- GPU with CUDA support
- [YOLOv8](https://github.com/ultralytics/yolov8)
- [SORT GitHub Repository](https://github.com/abewley/sort)
- Canva (for mask creation)

### Sample Outputs

#### **YOLOv8l (large) car counting Output**

![YOLOv8l Car_Counting_Output](https://github.com/Heniteg/Object-detection/blob/main/scripts/Running-Yolo/outputs/yolov8n_people_detection.png)

#### **Masking for the region of interest**

![Masking](https://github.com/Heniteg/Object-detection/blob/main/scripts/Running-Yolo/outputs/yolov8l_people_detection.png)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/car-counting-yolov8.git
   cd car-counting-yolov8


---

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

---

