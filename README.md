# Deepfake Detection System using EfficientNet + BiLSTM

A web-based **Deepfake Detection system** that allows users to upload videos and determine whether the media is **real or manipulated** using a hybrid **EfficientNet + BiLSTM deep learning model**.

This project combines **Computer Vision, Deep Learning, and Web Development** to provide an accessible tool for identifying manipulated video content.

---

# Overview

Deepfakes are synthetic media generated using deep learning techniques that manipulate faces in images or videos. These manipulations are becoming increasingly realistic and difficult to detect.

This project focuses on detecting deepfake videos using a **hybrid deep learning architecture** that combines:

- **EfficientNet-B0** for spatial feature extraction
- **Bi-directional LSTM (BiLSTM)** for temporal sequence analysis across video frames

The system provides a **Flask-based web interface** where users can upload video files and the model analyzes the content to classify it as **Real or Fake**.

---

# Features

- Upload videos for deepfake analysis
- Automatic frame extraction from uploaded videos
- Frame preprocessing and normalization
- EfficientNet-B0 for spatial feature extraction
- BiLSTM for temporal pattern detection
- Deep learning classification of real vs fake media
- Confidence score for prediction results
- Web interface built using Flask

---

# System Architecture

The system follows this pipeline:

1. User uploads a video through the web interface  
2. Video frames are extracted automatically  
3. Frames are preprocessed and resized to **224×224**  
4. EfficientNet-B0 extracts spatial features from each frame  
5. BiLSTM processes the temporal sequence of features  
6. Fully connected layers perform classification  
7. System displays prediction results and confidence score  

---

# Model Architecture

The Deepfake detection model uses a **hybrid EfficientNet + BiLSTM architecture** built using **PyTorch**.

### Input
- 16 frames per video
- Frame size: **224 × 224 × 3**

### Feature Extraction

**EfficientNet-B0**
- Pretrained convolutional neural network
- Extracts spatial features from each frame
- Output feature vector size: **1280**

### Temporal Learning

**Bi-directional LSTM**
- Hidden Size: 256
- Learns temporal dependencies across frames
- Bidirectional sequence learning

### Classification Layers

Fully Connected Layers:
- Dense layer (128 neurons)
- ReLU activation
- Dropout (0.5)
- Output layer (2 classes: **Real / Fake**)

---

# Training Configuration

The model was trained using:

- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Learning Rate:** 0.0001  
- **Epochs:** 30  
- **Sequence Length:** 16 frames  
- **Batch Size:** 4  

---

# Dataset

The model was trained using labeled frames extracted from real and fake videos.

### Dataset Structure

```
dataset/
├── real/
│   ├── video1/
│   ├── video2/
│
├── fake/
│   ├── video1/
│   ├── video2/
```

Each video folder contains extracted frames.

Images are resized to **224×224** and normalized before being passed to the model.

---

# Web Application

The project includes a **Flask-based web application** that allows users to interact with the trained model.

Users can:

- Upload video files (MP4)
- Automatically extract frames from videos
- Run deepfake analysis using the trained model
- View prediction results with confidence score

---

# Project Structure

```
lstm_deepfake_new/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── model.py
│   └── DFM_30700.pth
│
├── utils/
│   ├── predict.py
│   └── video_processing.py
│
├── templates/
│   ├── index.html
│   ├── upload.html
│   └── result.html
│
├── static/
│   ├── css/
│   ├── uploads/
│   └── frames/
```

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- Flask
- HTML / CSS / JavaScript
- NumPy
- Pillow

---

# Results

The system analyzes uploaded videos and outputs:

- Prediction result (**Real / Fake**)
- Confidence score of prediction
- Extracted frames from uploaded video

---

# Application Screenshots

## Home Page
(Add screenshot here)

## Upload Page
(Add screenshot here)

## Result Page
(Add screenshot here)

---

# Future Improvements

- Integrate face detection using MTCNN or RetinaFace
- Use larger datasets such as DFDC or FaceForensics++
- Improve real-time detection performance
- Deploy the system as a cloud-based AI service
- Add explainable AI visualization for predictions

---

# Author

**Meet Mistry**  
Computer Engineering Student  
Machine Learning & AI Enthusiast
