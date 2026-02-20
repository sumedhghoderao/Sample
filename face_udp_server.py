import cv2
import os
import numpy as np
import socket
import time

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_FOLDER = "images"
MODEL_PATH = "model.yml"

UDP_IP = "0.0.0.0"
UDP_PORT = 9999

FACE_BUFFER_TIME = 30  # seconds
CONFIDENCE_THRESHOLD = 70  # lower = stricter

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
# TRAIN FROM IMAGE FOLDER
# ----------------------------
def train_from_folder():

    faces = []
    labels = []

    print("📂 Loading images from folder...")

    for img_name in os.listdir(IMAGE_FOLDER):
        img_path = os.path.join(IMAGE_FOLDER, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected:
            face = gray[y:y+h, x:x+w]
            faces.append(face)
            labels.append(0)  # single person

    if len(faces) == 0:
        print("❌ No faces found in images folder")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    print(f"✅ Training complete using {len(faces)} images")

# ----------------------------
# RECOGNITION MODE
# ----------------------------
def recognize():

    global client_addr

    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Train first.")
        return

    recognizer.read(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    last_sent_time = 0

    print("🟢 Recognition started")

    while True:

        # ---- Check UDP client registration ----
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

                cv2.putText(frame, "YOU",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2)

                # 30 second buffer
                if client_addr and (current_time - last_sent_time) > FACE_BUFFER_TIME:
                    sock.sendto(b"FACE_RECOGNIZED", client_addr)
                    print("📡 FACE_RECOGNIZED sent")
                    last_sent_time = current_time

            else:
                cv2.putText(frame, "Unknown",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow("Jetson Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# MAIN MENU
# ----------------------------
while True:
    print("\nPress 1 → Train from images folder")
    print("Press 2 → Start recognition")
    print("Press q → Quit")

    choice = input("Enter choice: ")

    if choice == "1":
        train_from_folder()

    elif choice == "2":
        recognize()

    elif choice.lower() == "q":
        break
