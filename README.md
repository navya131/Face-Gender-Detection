# AI Face and Gender Recognition System using CNN and OpenCV

## Project Overview

The AI Face and Gender Recognition System is a computer vision application that identifies a person's name and predicts their gender using a webcam. The system combines OpenCV for face detection and recognition with a Convolutional Neural Network (CNN) for gender classification.

The application captures facial images, trains machine learning models, and performs real-time recognition through a webcam.

---

## Features

* Real-time face detection using OpenCV Haar Cascade.
* Face recognition for identifying registered users.
* Gender prediction using a CNN model.
* Webcam-based live detection.
* Automatic face dataset generation.
* Model training and testing support.
* User-friendly project structure.

---

## Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Pillow
* Scikit-learn

---

## Project Structure

AI-Face-Gender-Recognition/

├── dataset/

│   ├── gender/

│   │   ├── male/

│   │   └── female/

│   │

│   └── faces/

│       ├── Person1/

│       ├── Person2/

│       └── Person3/

│

├── models/

│   ├── gender_model.h5

│   ├── face_trainer.yml

│   └── labels.pkl

│

├── haarcascade/

│   └── haarcascade_frontalface_default.xml

│

├── capture_faces.py

├── train_gender_model.py

├── train_face_model.py

├── main.py

├── requirements.txt

└── README.md

---

## Installation

### Clone the Repository

```bash
git clone <repository_url>
cd AI-Face-Gender-Recognition
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

### Gender Dataset

Store gender images in:

dataset/gender/male/

dataset/gender/female/

### Face Recognition Dataset

Store person images in:

dataset/faces/PersonName/

Example:

dataset/faces/Navya/

dataset/faces/Ravi/

dataset/faces/Sai/

---

## Training the Gender Model

Run:

```bash
python train_gender_model.py
```

Output:

models/gender_model.h5

---

## Capturing Face Images

Run:

```bash
python capture_faces.py
```

Enter the person's name and capture approximately 30 facial images.

---

## Training Face Recognition Model

Run:

```bash
python train_face_model.py
```

Output:

models/face_trainer.yml

models/labels.pkl

---

## Running the Application

Run:

```bash
python main.py
```

The webcam will open and display:

Name: Detected Person

Gender: Male/Female

---

## Sample Output

Detected Person: Navya

Predicted Gender: Female

Detected Person: Ravi

Predicted Gender: Male

---

## Applications

* Smart Attendance Systems
* Security and Surveillance
* Access Control Systems
* Educational Projects
* Human-Computer Interaction

---

## Future Enhancements

* Age Detection
* Emotion Recognition
* Face Mask Detection
* Attendance Report Generation
* Cloud Database Integration

---

## Conclusion

This project demonstrates the integration of Deep Learning and Computer Vision techniques for real-time face recognition and gender classification. It provides a practical implementation of CNN models and OpenCV-based image processing suitable for academic and real-world applications.

---

### Author

Navya Botla

B.Tech Project – AI Face and Gender Recognition using CNN and OpenCV
