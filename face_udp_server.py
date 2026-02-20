import cv2
import os
import numpy as np
import socket
import time
import pickle

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_FOLDER = "images"
MODEL_PATH = "model.yml"
LABELS_PATH = "labels.pkl"

UDP_IP = "0.0.0.0"
UDP_PORT = 9999

FACE_BUFFER_TIME = 5
CONFIDENCE_THRESHOLD = 70

# ----------------------------
# UDP SETUP
# ----------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)

client_addr = None

# ----------------------------
# FACE SETUP
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

# ----------------------------
# TRAIN FUNCTION
# ----------------------------
def train_from_folders():

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    print("📂 Loading images...")

    for person_name in os.listdir(IMAGE_FOLDER):
        person_path = os.path.join(IMAGE_FOLDER, person_name)

        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in detected:
                face = gray[y:y+h, x:x+w]
                faces.append(face)
                labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        print("❌ No faces found.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    print(f"✅ Training complete. Trained {current_label} persons.")

# ----------------------------
# RECOGNITION FUNCTION
# ----------------------------
def recognize():

    global client_addr

    if not os.path.exists(MODEL_PATH):
        print("❌ Train first!")
        return

    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)

    cap = cv2.VideoCapture(0)

    last_sent_time = 0

    print("🟢 Recognition started")

    while True:

        # UDP registration
        try:
            data, addr = sock.recvfrom(1024)
            if data.decode() == "HELLO":
                client_addr = addr
                sock.sendto(b"ACK", client_addr)
                print("Client registered:", client_addr)
        except:
            pass

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        current_time = time.time()

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            if confidence < CONFIDENCE_THRESHOLD:

                name = label_map[label]

                cv2.putText(frame, name,
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2)

                if client_addr and (current_time - last_sent_time) > FACE_BUFFER_TIME:
                    message = f"FACE_RECOGNIZED:{name}"
                    sock.sendto(message.encode(), client_addr)
                    print(f"📡 Sent recognition for {name}")
                    last_sent_time = current_time

            else:
                cv2.putText(frame, "Unknown",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow("Jetson Multi-Person Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# MAIN MENU
# ----------------------------
while True:
    print("\nPress 1 → Train All Persons")
    print("Press 2 → Start Recognition")
    print("Press q → Quit")

    choice = input("Enter choice: ")

    if choice == "1":
        train_from_folders()

    elif choice == "2":
        recognize()

    elif choice.lower() == "q":
        break
