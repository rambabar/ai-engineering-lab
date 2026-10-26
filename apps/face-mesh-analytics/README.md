# Real-Time 468-Point Face Mesh & Biometric Geometry Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-orange)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive computer vision application for real-time facial landmark detection, 468-point 3D face mesh tessellation, iris tracking, and spatial geometry analysis built with **MediaPipe**, **OpenCV**, and **Streamlit**.

---

## 📽️ Live Demo

![Face Mesh & Iris Analytics Demo](assets/demo.gif)

> **High-Resolution Walkthrough**: A complete 1080p video demo is available at [`assets/demo.mp4`](assets/demo.mp4).

---

## ⚡ Key Capabilities

- **468-Point 3D Face Mesh**: Full facial surface reconstruction with sub-millimeter depth estimation and 3D coordinate mapping.
- **Iris Landmark & Gaze Tracking**: Refined eye contour and iris center localization for attention and liveness indicators.
- **Contour & Geometric Segmentation**: Real-time extraction of lips, eyebrows, eye sockets, and oval facial boundaries.
- **Sub-15ms Local Inference**: Lightweight, edge-optimized inference pipeline running fully CPU-friendly on standard hardware.
- **Customizable Overlays**: Configurable mesh density, landmark colors, line thickness, and detection confidence thresholds.

---

## 🏗️ Architecture

```
                       ┌─────────────────────────┐
                       │  Input Image / Webcam   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  RGB Color Conversion   │
                       │     & Preprocessing     │
                       └────────────┬────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       ┌────────────────────────┐      ┌────────────────────────┐
       │   MediaPipe Detection  │      │   MediaPipe Face Mesh  │
       │    (BBox + Keypoints)  │      │     (468 Landmarks)    │
       └────────────┬───────────┘      └────────────┬───────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │   Geometric Rendering   │
                       │   & Coordinate Metrics  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Interactive UI Analytics│
                       │   (Streamlit Dashboard) │
                       └─────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/rambabar/face-mesh-analytics.git
cd face-mesh-analytics
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Create venv
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate
# Or on Linux/macOS:
# source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Repository Structure

```
face-mesh-analytics/
├── assets/
│   ├── demo.gif               # Optimized UI preview GIF
│   └── demo.mp4               # Full 1080p demo walkthrough
├── app.py                     # Main Streamlit & MediaPipe pipeline
├── requirements.txt           # Version-pinned dependencies
├── face_test1.jpg             # Sample test image 1
├── face_test2.jpg             # Sample test image 2
├── face_test3.jpg             # Sample test image 3
├── .gitignore
└── README.md
```

---

## ⚙️ Technical Highlights

| Component | Specification |
| :--- | :--- |
| **Pipeline Core** | MediaPipe Face Mesh Solutions API (`0.10.14`) |
| **Image Processing** | OpenCV 4.x & NumPy Vectorized Coordinates |
| **GUI Framework** | Streamlit |
| **Landmark Count** | 468 Dense 3D Points + Optional 10 Iris Keypoints |
| **Inference Mode** | Real-time static image & batch video frame processing |

---

## 👤 Author

**Ramchandra Babar**  
- Portfolio: [ram-portfolio-green.vercel.app](https://ram-portfolio-green.vercel.app/)  
- LinkedIn: [linkedin.com/in/rambabar](https://www.linkedin.com/in/rambabar)  
- GitHub: [@rambabar](https://github.com/rambabar)
