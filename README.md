# YOLOv8 Object Detection with Webcam

This project demonstrates real-time object detection using **YOLOv8** models on a live webcam feed, focusing on detecting people and phones. The experiment was conducted within a Python virtual environment on a Linux system to ensure reproducibility and isolation from global dependencies.

## Project Overview

In this experiment, different **YOLOv8** weights, including **YOLOv8n** (nano), were used to detect objects in real-time from a webcam feed. The goal was to achieve fast and accurate detection suitable for interactive applications, like security monitoring and human-computer interaction.

The object detection was executed using the script `yoloWithWebcam.py`, which captures frames from the webcam and applies YOLO models to detect objects like people and phones.

> **Note:** The models were evaluated in real-time. Check out the results below for more details.

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
    - On **Windows**:
        ```bash
        venv\Scripts\activate
        ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## YOLOv8 Model Used

| Model     | Size  | Speed | Accuracy | Use Case              |
| --------- | ----- | ----- | -------- | --------------------- |
| YOLOv8n   | Small | Fast  | Moderate | Real-time detection   |

### Sample Outputs

#### **YOLOv8n (Nano) Detection Output**
This model is designed for speed, making it ideal for real-time detection. It successfully detected the majority of people and phones but may miss smaller or more complex objects in challenging conditions.

![YOLOv8n Person Detection Output](https://github.com/Heniteg/Object-detection/blob/yolo-webcam-enhancements/scripts/Yolo-With-Webcam/photos-and-videos/person-detection-with-webcam.png)

In this example, the YOLOv8n model accurately detects a person in real-time with minimal latency.

![YOLOv8n Phone Detection Output](https://github.com/Heniteg/Object-detection/blob/yolo-webcam-enhancements/scripts/Yolo-With-Webcam/photos-and-videos/phone-detection-with-webcam.png)

![Video Sample] (https://github.com/Heniteg/Object-detection/blob/yolo-webcam-enhancements/scripts/Yolo-With-Webcam/photos-and-videos/yolo-object-detection-using-webcam.mp4)

Here, the model detects a phone in the user's hand, demonstrating its ability to recognize smaller objects effectively.

---

## How to Use

To replicate the experiment:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Heniteg/Object-detection.git
    cd Object-detection
    git checkout yolo-webcam-enhancements
    ```

4. **Run the object detection**:
    - Modify the model weights inside `yoloWithWebcam.py`:
        ```python
        model = YOLO("../../Yolo-Weights/yolov8n.pt")  # You can switch to other YOLOv8 weights if available
        ```

    - Then run the script:
        ```bash
        python scripts/Yolo-With-Webcam/yoloWithWebcam.py
        ```

---

## Results

- **YOLOv8n** offers a good balance of speed and accuracy, making it ideal for real-time applications where quick feedback is essential.
- The model can reliably detect larger objects like people and smaller objects like phones, but its performance may vary depending on the environment and lighting conditions.

---

## Additional Information

- **YOLOv8 Documentation**: [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- **Environment**: Python 3.10, Linux operating system

---
