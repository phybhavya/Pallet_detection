# Pallet_detection ROS2 and YOLOv11**

This project detects pallets in a factory workshop using YOLOv11 integrated with ROS2 for real-time processing. It supports the **ZED 2i stereo camera** for advanced depth-based applications and can also work with a standard camera by modifying the subscribed topic.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
   - [System Requirements](#system-requirements)
   - [Installation](#installation)
3. [Commands to Run the Project](#commands-to-run-the-project)
---

## **Project Overview**
The system processes video feeds to detect pallets using a YOLOv11 model deployed with ROS2. The following features are supported:
- **ZED 2i Stereo Camera**: Leverages high-quality image capture and depth perception.
- **Standard Camera Support**: Allows flexibility for simpler setups by changing the subscribed topic.

---

## **Setup Instructions**

### **System Requirements**
- **Operating System**: Ubuntu 22.04 / Windows Subsystem for Linux (WSL) 2
- **ROS2 Distribution**: Humble/
- **Python**: 3.10 or later
- **Dependencies**:
  - `ultralytics`, `opencv-python`

---

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/phybhavya/Pallet_detection.git
   cd Pallet_detection

2. Install ROS2 dependencies:
   ```bash
        sudo apt update && sudo apt install -y python3-colcon-common-extensions

3. Build ROS2 Workspace
   ```bash
        colcon build
        source install/setup.bash
4. Install Python dependencies:
    ```bash
        pip install ultralytics opencv-python

---


### **Commands to Run the Project**
1. Run the yolo node with detection and segmentation
   ```bash
   	ros2 run camera_interface yolo_model
  
  if you dont have zed2i camera available you can use your device camera, some of the lines needs to be commented, explained in the code itself	


---
### **Future scope**
1. export the model to onnx so it js ready for edge deployment
2. solve the issues of onnx library on device
3. Dockerize the whole package for edge deployment eith having necessary nvidia drivers in it
4. Improve model accuracy by training on a bigger data set and annotate it better
