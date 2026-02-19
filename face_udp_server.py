import cv2
import socket
import time

# ----------------------------
# UDP CONFIG
# ----------------------------
UDP_IP = "0.0.0.0"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)

print("🟢 UDP Server running on port 9999")

client_addr = None

# ----------------------------
# FACE DETECTION CONFIG
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

FACE_BUFFER_TIME = 30   # seconds
last_face_time = 0

print("📷 Camera started, waiting for client...")

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:

    # ---- Listen for client registration ----
    try:
        data, addr = sock.recvfrom(1024)
        if data.decode() == "HELLO":
            client_addr = addr
            sock.sendto(b"ACK", client_addr)
            print(f"✅ Client registered: {client_addr}")
    except socket.timeout:
        pass

    # ---- Camera frame ----
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    current_time = time.time()

    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Buffer logic
