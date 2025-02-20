# Real-Time Object Detection for Autonomous Driving with YOLO & Waymo Dataset

## Project Overview
This project focuses on real-time object detection for autonomous driving using YOLOv8, trained on the Waymo Open Dataset. The goal is to accurately detect:

- **Vehicles**
- **Pedestrians**
- **Cyclists**

---

## Why Use Google Cloud Storage (GCS) Instead of Downloading Data Locally?
One of the biggest challenges I faced while working with the **Waymo Open Dataset** is its size. Downloading and storing it locally is just not practical.

Instead of manually downloading, extracting, and managing huge files, this project streams the dataset directly from the official Google Cloud Storage (GCS) bucket. This means:

- **No need for large local storage** – All images and labels are read dynamically.  
- **Faster training setup** – No waiting for massive downloads.  
- **Always up-to-date** – We can always use the latest dataset version.  
- **Scalable & flexible** – The model can train on larger datasets without storage constraints.  

This approach makes training more efficient by allowing the model to access only the data it needs, when it needs it, without having to worry about local storage capacity.

---

## How This Project Works (High-Level Breakdown)

### **Loading Data from GCS**
Instead of downloading individual images, the labels and images are read in the TFRecords format directly from Google Cloud Storage.

The data loader:
- Connects to GCS and reads raw dataset files.
- Extracts camera images and bounding box labels from TFRecords.
- Converts everything into a format YOLOv8 can understand.

This step removes unnecessary data storage.

---

### **Preparing Data for YOLOv8**
Once images and labels are extracted, they need to be formatted for YOLOv8. Since Waymo stores bounding boxes in absolute pixel values, but YOLO expects normalized values instead, some preprocessing steps are required.

The preprocessing includes:
- **Image resizing** – Since YOLOv8 expects 640x640 images, we resize frames from Waymo’s original 1920x1080 resolution.
- **Bounding box conversion** – Convert pixel coordinates into YOLO’s required format (`class_id x_center y_center width height`).
- **Class mapping** – Assign vehicles (0), pedestrians (1), and cyclists (2).

This preprocessing ensures that YOLO can correctly detect objects in an urban driving scene.

---

### **Training YOLOv8**
Once the dataset is formatted, we train **YOLOv8-Nano**.

The training:
- Runs for **50 epochs** to allow the model to learn patterns.
- Uses **80% of the dataset for training** and **20% for validation**.
- Learns to predict bounding boxes and classify objects accurately.

The model is optimized with gradient descent to help adjust internal parameters and improve detection accuracy over time.

---

## Model Performance & Achieved Metrics
After training, the model was tested on urban driving scenes and achieved:

- **Mean Average Precision:** 76.5%  
- **Inference Speed:** 6 milliseconds per frame  
- **Dataset Size Processed:** 100K frames  
- **Classes Detected:** Vehicles, Pedestrians, Cyclists  

---

## Running Inference on New Images
Once trained, the model can be used to detect objects in real-world driving scenes. The process:

1. Load the trained YOLOv8 model.  
2. Pass a new image into the model.  
3. The model predicts objects and draws bounding boxes around detected cars, pedestrians, and cyclists.  

---

## What Could Be Improved?
Even though the model works well, it could be made more efficient.

### **Challenges Faced**
#### 1. **Streaming Data Introduces Latency**  
- Since we read from GCS, there’s a small delay while fetching images.  
- A possible fix is caching frequently accessed images locally (though this may be more storage-intensive).  

#### 2. **Higher Image Resolution**  
- Waymo’s images are **1920x1080**, but **640x640** images were used for better efficiency.  
- Higher-resolution input could help improve detection accuracy.  

#### 3. **Class Imbalance in the Dataset**  
- The dataset has **more vehicles than pedestrians and cyclists**.  
- A weighted loss function or additional datasets from the Waymo training library could balance the training data.  

#### 4. **Use a Larger YOLO Model**  
- **YOLOv8-Small or YOLOv8-Medium** could improve accuracy.  

#### 5. **Enhance Data Augmentation**  
- Adding **motion blur, lighting variations, and occlusions** could improve real-world performance.  

---

## Technologies Used
This project brings together multiple technologies for deep learning, cloud storage, and dataset processing:

- **YOLOv8** – Object detection model  
- **Google Cloud Storage (GCS)** – Streaming dataset without local downloads  
- **PyTorch** – Model training framework  
- **Bazel** – Used for compiling the Waymo dataset efficiently  
- **TensorFlow** – Used for processing TFRecords  

---

## Waymo License
This code repository (excluding `src/waymo_open_dataset/wdl_limited` folder) is licensed under the **Apache License, Version 2.0**. The code appearing in `src/waymo_open_dataset/wdl_limited` is licensed under terms appearing therein.

The **Waymo Open Dataset** itself is licensed under separate terms. Please visit [Waymo Open Dataset License](https://waymo.com/open/terms/) for details.

Code located in each of the subfolders located at `src/waymo_open_dataset/wdl_limited` is licensed under:

- **(a) BSD 3-clause copyright license**  
- **(b) An additional limited patent license**  

Each limited patent license is applicable **only** to code under the respective `wdl_limited` subfolder and is licensed for use **only** with the use case laid out in such a license in connection with the **Waymo Open Dataset**, as authorized by and in compliance with the **Waymo Dataset License Agreement for Non-Commercial Use**.

See `wdl_limited/camera/`, `wdl_limited/camera_segmentation/`, and `wdl_limited/sim_agents_metrics/` respectively for details.

---
