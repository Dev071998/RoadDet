
# 🛣️ Lane Detection System using OpenCV & Hough Transform

This project implements a real-time lane detection pipeline using Python and OpenCV. It processes road images or video frames, enhances them using Gaussian Blur and Canny edge detection, and detects lane lines using the Probabilistic Hough Line Transform. The final result overlays detected lane lines on the original video.

---

## 📌 Features

- Real-time processing of road images
- Lane detection using Hough Transform
- Moving average filtering for line stability
- Sharpness enhancement for better accuracy
- Exports the output video with overlaid lane lines
- Auto video playback after processing

---

## 🛠️ Setup Instructions

### ✅ Prerequisites

Make sure you have Python installed. This project uses:

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy

### 🔧 Installation

1. **Clone the repository**:

```bash
git clone "git@github.com:Dev071998/RoadDet.git"
cd RoadDet
```

2. **Create a virtual environment (optional but recommended)**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


> If you don't have a `requirements.txt` yet, create one with:
> ```txt
> opencv-python
> numpy
> ```

---


## 🚀 How to Run

 ```bash
gcc main.c
```
1. Place road image sequence in the `photos` folder.
2. Run the script:

```bash
python lane_detection.py
```

3. The output video will be saved in the `processed_videos/` folder.
4. After processing, the final video will automatically play. Press `Q` to exit early.

---

## 🖼️ Sample Output

![lane_detection_demo](https://github.com/user-attachments/assets/41d3ecc3-dc0d-48fc-bc04-a003416d1cc7)

---

## 🧠 Technical Overview

- **Gaussian Blur**: Reduces noise before edge detection.
- **Canny Edge Detection**: Extracts edges from the image.
- **Region of Interest Masking**: Focuses only on the road area.
- **HoughLinesP**: Detects lines in the masked edge image.
- **Moving Average Smoothing**: Stabilizes detection across frames.

---

## 📌 TODOs

- [ ] Add lane curvature estimation

---

## 📜 License

NA

---

## 🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---
