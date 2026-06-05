import cv2
import os

person_name = input("Enter Person Name: ")

save_path = os.path.join("dataset", person_name)
os.makedirs(save_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = len(os.listdir(save_path))

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

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face = frame[y:y+h, x:x+w]

        cv2.putText(
            frame,
            "Press C to Capture",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Dataset Collection", frame)

    key = cv2.waitKey(1)

    if key == ord('c') and len(faces) > 0:

        count += 1

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (128, 128))

        cv2.imwrite(
            os.path.join(save_path, f"{count}.jpg"),
            face
        )

        print(f"Saved Image {count}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()