
# 🎾 SADATrack: Advanced AI Tennis Analysis

**SADATrack** is a comprehensive computer vision system designed to analyze tennis match footage. Going beyond simple object tracking, this system employs a multi-stage pipeline to generate professional-grade match statistics, including real-time shot classification, player speed analysis, and a 2D tactical mini-map.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## 🚀 Key Features

Unlike standard tracking solutions, this project integrates **Action Recognition** and **Physics Mapping** to provide deep insights:

* **Player Tracking:** Uses **YOLO** to detect and track players, automatically filtering out non-game personnel (ball boys, judges) based on court positioning.
* **Ball Trajectory & Bounce Detection:** Implements **TrackNet** for high-precision ball tracking and a custom **CatBoost** regression model to detect exact bounce locations.
* **3D Action Recognition:** A custom **Hybrid Directed Graph Convolutional Network (HDGCN)** analyzes skeletal movements to classify shots (e.g., *Forehand Flat, Backhand Slice, Kick Service*).
* **Tactical Mini-Map:** Utilizes Homography and **ResNet50** court line detection to map video pixels to real-world meters, generating a top-down tactical view.
* **Performance Metrics:** Calculates real-time player speed, distance covered, and ball shot speed.

## 🛠️ The Pipeline

The system processes video through several distinct modules:

1.  **Court Line Detection:** Finds key points (corners, service lines) to establish a coordinate system.
2.  **Object Tracking:** Detects players and the ball across frames.
3.  **Pose Estimation:** Extracts skeletal keypoints from players using YOLO-Pose.
4.  **Action Classification:** Feeds skeletal time-series data into the **HDGCN** model to identify the stroke type.
5.  **Statistical Integration:** Merges tracking data with the mini-map to compute physical metrics.
6.  **Visualization:** Overlays bounding boxes, skeletons, stats, and the mini-map onto the original video.

## 📂 Project Structure

```text
CV_project/
├── action_recognition/    # (New) HDGCN model & shot classification logic
├── court_line_detector/   # ResNet50 model for court keypoints
├── mini_court/            # Homography and coordinate transformation
├── trackers/              # YOLO player tracking & TrackNet ball tracking
├── utils/                 # Video processing and drawing utilities
└── main.py                # Main pipeline entry point

```

## 🤝 Acknowledgments & Evolution

This project was originally forked from [tennis_analysis by abdullahtarek](https://github.com/abdullahtarek/tennis_analysis).

While the original repository provided an excellent foundation for object tracking and court detection, **this project has been significantly overhauled and expanded.** Major deviations include:

* **Integrated Action Recognition:** Added a completely new module (`action_recognition/`) utilizing Graph Convolutional Networks (HDGCN) to understand *what* the players are doing, not just *where* they are.
* **Advanced Bounce Detection:** Replaced heuristic bounce detection with a trained **CatBoost** model for higher accuracy.
* **Shot Prediction Logic:** Implemented logic to correlate ball position with player handedness to refine shot classification.
* **Performance Optimization:** Refined the processing loop for better handling of long rallies.
