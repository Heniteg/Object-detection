YOLOv8 Object Detection Experiment
This project explores the capabilities of YOLOv8 (You Only Look Once) object detection models in identifying people and cars from images. The experiment leverages two different YOLOv8 weights—YOLOv8n (nano) and YOLOv8l (large)—to compare performance in terms of speed and accuracy. The entire setup was conducted in a Python virtual environment on a Linux system to ensure reproducibility and isolation from global dependencies.

Project Overview
This experiment aims to highlight the trade-offs between real-time performance and detection accuracy when using different YOLOv8 models. By running the same set of images through YOLOv8n and YOLOv8l, we can observe how each model responds to the task of detecting both people and cars in various scenes.

The experiment was implemented using the script yolloBasics.py, which processes input images using the specified YOLO model weights and outputs the detected objects with bounding boxes.

Virtual Environment Setup
The project is contained within a Python virtual environment to ensure compatibility and isolation from system-wide packages. All dependencies are managed through requirements.txt, allowing easy setup on any machine.

Key steps:

Virtual environment creation and activation.
Installation of YOLO dependencies.
Execution of object detection with YOLOv8 models.
YOLOv8 Models Used
Model	Size	Speed	Accuracy	Use Case
YOLOv8n	Small	Fast	Moderate	Real-time detection
YOLOv8l	Large	Slower	High	High-accuracy tasks
Sample Outputs
YOLOv8n (Nano) Detection Output:
This model is designed for speed, making it ideal for real-time detection, though it sacrifices some accuracy in detecting smaller or more complex objects. In this experiment, YOLOv8n successfully detected the majority of people and cars but missed some details.


YOLOv8l (Large) Detection Output:
With a focus on accuracy, YOLOv8l demonstrated superior object detection capabilities, identifying all people and cars in the images with high precision. However, this came at the cost of slower inference times.


Results
YOLOv8n: Best suited for applications requiring high-speed, real-time object detection, though it may miss smaller objects or complex scenarios.
YOLOv8l: Ideal for situations where accuracy is paramount, albeit with a trade-off in processing speed.
How to Use
Cloning the Repository
To replicate this experiment on your local machine:

git clone https://github.com/Heniteg/Object-detection.git
cd Object-detection

Setting Up the Virtual Environment
Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Installing Dependencies
Install all required packages from the requirements.txt file:
pip install -r requirements.txt

Running the Object Detection
To run the detection script, you can modify the model weights directly inside the yolloBasics.py script to either YOLOv8n or YOLOv8l (or any other YOLOv8 model) by adjusting the path to the model's weights. After modifying the script, execute it using the virtual environment's Python interpreter.



