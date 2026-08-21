# Object Tracking Using Image Processing

Vehicle detection and object tracking in video streams using OpenCV and Python.

## 📌 Features
- **Background Subtraction**: Uses Gaussian Mixture-based Background/Foreground Segmentation (`MOG2`) to extract moving objects.
- **Euclidean Distance Tracking**: Assigns and maintains unique persistent IDs for detected objects across video frames.
- **Interactive Streamlit Web App**: Real-time mask generation and parameter tuning interface.

## 📁 Project Structure
- `1. Video capture.py`: Basic video reading and frame display with OpenCV.
- `2. White mask.py`: Motion mask extraction using `createBackgroundSubtractorMOG2`.
- `main.py`: Full pipeline with Region of Interest (ROI) selection, contour detection, and Euclidean distance tracking.
- `tracker.py`: `EuclideanDistTracker` class calculating object centroids and matching IDs.
- `app.py`: Streamlit web dashboard for interactive tracking and mask analysis.
- `highway.mp4`: Sample traffic video dataset.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Main Tracker
```bash
python main.py
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```
