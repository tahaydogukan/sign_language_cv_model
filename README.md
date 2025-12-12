# 🖐️ Sign Language to Speech – MediaPipe & LSTM

This project is a **real-time sign language recognition system** that detects hand and body movements from a camera, classifies them using a deep learning model.

https://github.com/user-attachments/assets/793e85af-dbbd-4590-b02a-3ac968036f37

## 🚀 Project Overview

The system works in real time and follows this pipeline:

```
Camera → MediaPipe Holistic → Landmark Extraction
       → LSTM Model → Sign Classification
```

Instead of using single images, each sign is treated as a **sequence of frames**, allowing the model to understand motion and temporal patterns.

---

## 🧠 How It Works

### 1️⃣ Landmark Extraction (MediaPipe)

* Uses **MediaPipe Holistic**
* Extracts:

  * Right hand landmarks (21 points)
  * Left hand landmarks (21 points)
  * Body pose landmarks (33 points)
* Each frame is converted into a numerical feature vector

### 2️⃣ Sequence-Based Learning (LSTM)

* Each sign is recorded as **30 consecutive frames**
* These sequences are fed into an **LSTM (Long Short-Term Memory)** neural network
* LSTM learns the temporal structure of gestures rather than static poses

### 3️⃣ Real-Time Inference

* Live camera feed is processed frame by frame
* The last 30 frames are continuously evaluated
* The model predicts the most likely sign

## 🧩 Supported Signs (Example)

```text
MERHABA
EVET
HAYIR
```

## 🛠️ Technologies Used

* **Python**
* **MediaPipe (Holistic)**
* **TensorFlow / Keras**
* **LSTM Neural Networks**
* **OpenCV**
* **NumPy***
* **scikit-learn**


---

## 📊 Data Format

Each recorded sample is saved as a NumPy file:

```python
(30, 258)
```

* `30` → number of frames (sequence length)
* `258` → feature vector per frame (hands + pose landmarks)

---

## ▶️ How to Run

### 1️⃣ Create & activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn
```

### 3️⃣ Collect data

```bash
python collect_data.py
```

### 4️⃣ Train the model

```bash
python train_lstm.py
```

### 5️⃣ Run real-time inference

```bash
python realtime_inference.py
```

---

## 🎯 Key Learnings

* Temporal data is critical for gesture recognition
* LSTM models outperform single-frame classifiers for motion-based tasks
* MediaPipe provides a powerful and efficient way to extract body landmarks


## 📌 Disclaimer

This project is for **educational and experimental purposes**.
It is not intended to replace professional sign language interpreters.




