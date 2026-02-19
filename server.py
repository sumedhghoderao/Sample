import cv2
import os
import numpy as np
import socket
import time
import pickle

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_DIR = "images"
MODEL_PATH = "model.yml"
LABELS_PATH = "labels.pkl"

SERVER_IP = "0.0.0.0"
SERVER_PORT = 9999

FACE_BUFFER_TIME = 30  # seconds

# -----------------------------
# Face detector & recognizer
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# -----------------------------
# Train function
# -----------------------------
def train_faces():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(IMAGE_DIR):
        person_path = os.path.join(IMAGE_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            detected = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in detected:
                face = img[y:y+h, x:x+w]
                faces.append(face)
                labels.append(current_label)

        current_label += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    print("✅ Training completed and model saved.")

# -----------------------------
# Server socket
# -----------------------------
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_IP, SERVER_PORT))
server_socket.listen(1)
print("🟢 Waiting for client...")
client_socket, addr = server_socket.accept()
print(f"🔗 Client connected: {addr}")

# -----------------------------
# Load model
# -----------------------------
def load_model():
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "rb") as f:
        return pickle.load(f)

# -----------------------------
# Face detection
# -----------------------------
def detect_faces():
    label_map = load_model()
    cap = cv2.VideoCapture(0)

    last_detection_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        current_time = time.time()

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)

            if confidence < 80:
                name = label_map[label]
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 30-second buffer logic
                if current_time - last_detection_time > FACE_BUFFER_TIME:
                    client_socket.sendall(b"FACE_DETECTED")
                    print("📡 Signal sent to client")
                    last_detection_time = current_time

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Face Detection Server", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# MAIN MENU
# -----------------------------
while True:
    print("\nPress 1 → Train Faces")
    print("Press 2 → Start Detection")
    print("Press q → Quit")

    key = input("Enter choice: ")

    if key == "1":
        train_faces()
    elif key == "2":
        detect_faces()
    elif key.lower() == "q":
        break
