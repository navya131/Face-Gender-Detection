import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load models
face_model = load_model("models/face_model.h5")
gender_model = load_model("models/gender_model.h5")

# Load class names
with open("models/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face, (128, 128))

        face_input = face.astype("float32") / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        # -------------------
        # FACE RECOGNITION
        # -------------------

        face_pred = face_model.predict(
            face_input,
            verbose=0
        )

        confidence = np.max(face_pred) * 100
        person_idx = np.argmax(face_pred)

        if confidence < 70:
            person_name = "Unknown"
        else:
            person_name = class_names[person_idx]

        # -------------------
        # GENDER PREDICTION
        # -------------------

        gender_pred = gender_model.predict(
            face_input,
            verbose=0
        )

        score = gender_pred[0][0]

        if score >= 0.5:
            gender = "Female"
            gender_conf = score * 100
        else:
            gender = "Male"
            gender_conf = (1 - score) * 100

        # -------------------
        # DRAW
        # -------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{person_name} ({confidence:.1f}%)",
            (x, y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Gender: {gender} ({gender_conf:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Face Recognition",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()