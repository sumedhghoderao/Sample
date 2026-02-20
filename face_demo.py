import cv2
import os
import numpy as np
import pickle

# ============================
# CONFIG
# ============================

IMAGE_FOLDER = "images"
MODEL_PATH = "model.yml"
LABELS_PATH = "labels.pkl"

CONFIDENCE_THRESHOLD = 58

# Map person name → fullscreen image
IMAGE_MAP = {
    "sumedh": "IMAGE1.png",
    "vijay": "IMAGE2.png"
}

# ============================
# FACE SETUP
# ============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

# ============================
# TRAIN FUNCTION
# ============================

def train_from_folders():

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    print("📂 Training from images folder...")

    for person_name in os.listdir(IMAGE_FOLDER):

        person_path = os.path.join(IMAGE_FOLDER, person_name)

        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name
        print(f"➡ Training: {person_name}")

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

    print("✅ Training complete")

# ============================
# RECOGNITION FUNCTION
# ============================

def recognize():

    if not os.path.exists(MODEL_PATH):
        print("❌ Train first!")
        return

    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)

    cap = cv2.VideoCapture(0)

    current_displayed_person = None
    window_name = "Fullscreen Alert"

    print("🟢 Recognition started (press q to quit)")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_person = None

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            if confidence < CONFIDENCE_THRESHOLD:

                name = label_map[label]
                detected_person = name

                cv2.putText(frame,
                            f"{name} ({int(confidence)})",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2)

            else:
                cv2.putText(frame,
                            "Unknown",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # ============================
        # FULLSCREEN IMAGE LOGIC
        # ============================

        if detected_person and detected_person != current_displayed_person:

            if detected_person in IMAGE_MAP:

                image_path = IMAGE_MAP[detected_person]

                if os.path.exists(image_path):

                    img = cv2.imread(image_path)

                    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN
                    )

                    cv2.imshow(window_name, img)

                    current_displayed_person = detected_person

        # Show camera preview
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================
# MAIN MENU
# ============================

while True:
    print("\nPress 1 → Train")
    print("Press 2 → Start Recognition")
    print("Press q → Quit")

    choice = input("Enter choice: ")

    if choice == "1":
        train_from_folders()

    elif choice == "2":
        recognize()

    elif choice.lower() == "q":
        break