# YOLOv8 Object Detection Experiment

This project demonstrates object detection using **YOLOv8** models on images containing people and cars. The experiment was conducted within a Python virtual environment on a Linux system to ensure reproducibility and isolation from global dependencies.

## Project Overview

In this experiment, different **YOLOv8** weights, including **YOLOv8n** (nano) and **YOLOv8l** (large), were used to detect objects in images. The goal was to compare performance in terms of speed and accuracy for different models on the same dataset.

The object detection was executed using the script `yolloBasics.py`, which processes input images and applies YOLO models to detect objects like people and cars.

> **Note:** The models were trained and tested on a variety of images. Check out the results below for more details.

---

## Virtual Environment Setup

The project uses a Python virtual environment to ensure compatibility across different environments. Dependencies are specified in the `requirements.txt` file, making it easy for others to replicate the setup.

### Steps:
1. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    ```

2. **Activate the virtual environment**:
    - On **Linux/macOS**:
        ```bash
        source venv/bin/activate
        ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## YOLOv8 Models Used

| Model     | Size  | Speed | Accuracy | Use Case              |
| --------- | ----- | ----- | -------- | --------------------- |
| YOLOv8n   | Small | Fast  | Moderate | Real-time detection    |
| YOLOv8l   | Large | Slower| High     | High-accuracy tasks    |

### Sample Outputs

#### **YOLOv8n (Nano) Detection Output**
This model is designed for speed, making it ideal for real-time detection, though it sacrifices some accuracy in detecting smaller or more complex objects. In this experiment, YOLOv8n successfully detected the majority of people and cars but missed some details.

![YOLOv8n People_Detection_Output](https://github.com/Heniteg/Object-detection/blob/main/scripts/Running-Yolo/outputs/yolov8n_people_detection.png)
![YOLOv8n Cars_Detection_Output](https://github.com/Heniteg/Object-detection/blob/main/scripts/Running-Yolo/outputs/yolov8n_cars_detection.png)

#### **YOLOv8l (Large) Detection Output**
With a focus on accuracy, YOLOv8l demonstrated superior object detection capabilities, identifying all people and cars in the images with high precision. However, this came at the cost of slower inference times.

![YOLOv8l People_Detection_Output](https://github.com/Heniteg/Object-detection/blob/main/scripts/Running-Yolo/outputs/yolov8l_people_detection.png)
![YOLOv8l Cars_Detection_Output](https://github.com/Heniteg/Object-detection/blob/main/scripts/Running-Yolo/outputs/yolov8l_cars_detection.png)

---

## How to Use

To replicate the experiment:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Heniteg/Object-detection.git
    cd Object-detection
    ```

2. **Set up the virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the object detection**:
    - Modify the model weights inside `yolloBasics.py`:
        ```python
        model = YOLO("yolov8n.pt")  # You can switch to yolov8l.pt or any other model
        ```

---

## Results

- **YOLOv8n** is ideal for real-time applications, but it may miss smaller or more complex objects.
- **YOLOv8l** offers higher accuracy and can detect smaller objects with precision, but it comes with a trade-off in speed.

---



# YOLOv8 Object Detection on Video Feeds

This project demonstrates object detection using **YOLOv8** models on video feeds. The experiment was conducted using pre-recorded videos to detect objects like people, bikes, cars, and other common objects. The project aims to showcase the capabilities of the YOLOv8 model in recognizing multiple objects in dynamic video scenes.

## Project Overview

In this experiment, different **YOLOv8** weights, including **YOLOv8n** (nano) and **YOLOv8l** (large), were used to detect objects in videos. The goal was to achieve high accuracy in identifying various objects in diverse environments.

The object detection was executed using the script `yoloWithVideo.py`, which processes input videos and applies YOLO models to detect objects such as people, bikes, and cars.

> **Note:** The models were trained and tested on a variety of video scenarios. Check out the results below for more details.

---

### Sample Outputs

#### **Person Detection in Video**
The YOLOv8n model was used to detect people in a video feed. Despite being optimized for speed, it accurately identifies multiple people in the frame.

![Person Detection Video](https://github.com/Heniteg/Object-detection/blob/main/scripts/Running-Yolo/outputs/people.gif)

In this video, the model demonstrates its ability to detect people in various positions and movements.

#### **Bike Detection in Video**
The YOLOv8l model was used to detect bikes in a busy urban environment. Its high accuracy enables it to distinguish between bikes and other objects.

![Bike Detection Video](https://github.com/Heniteg/Object-detection/blob/main/scripts/Running-Yolo/outputs/motorbikes.gif)

This video showcases the model's precision in identifying bikes in complex scenes.

#### **Car Detection in Video**
The model accurately detects cars in a highway scenario, highlighting its use case for traffic monitoring and autonomous driving research.

![Car Detection Video](https://github.com/Heniteg/Object-detection/blob/main/scripts/Running-Yolo/outputs/cars.gif)

This video demonstrates the model's performance in identifying moving cars with high confidence.

---

## How to Use

To replicate the experiment:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Heniteg/Object-detection.git
    cd Object-detection
    git checkout yolo-detection-experiment
    ```

2. **Run the object detection on video**:
    - Modify the model weights and input video path inside `video_detection.py`:
        ```python
        model = YOLO("weights/yolov8n.pt")  # You can switch to yolov8l.pt or any other model
        video_path = "path_to_your_video/video.mp4"
        ```

    - Then run the script:
        ```bash
        python scripts/Yolo-With-Webcam/yoloWithWebcam.py
        ```

---

