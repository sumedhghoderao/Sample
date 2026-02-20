import socket
import cv2
import os
import time
import threading

SERVER_IP = "172.20.10.15"
SERVER_PORT = 9999

IMAGE_MAP = {
    "Sumedh": "IMAGE1.png",
    "Vijay": "IMAGE2.png"
}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.5)

def send_hello():
    while True:
        try:
            sock.sendto(b"HELLO", (SERVER_IP, SERVER_PORT))
        except:
            pass
        time.sleep(10)

threading.Thread(target=send_hello, daemon=True).start()

print("Client running (auto-reconnect enabled)")

while True:
    try:
        data, _ = sock.recvfrom(1024)
        message = data.decode()

        if message.startswith("FACE_RECOGNIZED:"):

            name = message.split(":")[1]
            print(f"{name} recognized")

            if name in IMAGE_MAP:
                image_file = IMAGE_MAP[name]

                if os.path.exists(image_file):
                    img = cv2.imread(image_file)

                    window_name = "Recognition Alert"
                    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN
                    )

                    cv2.imshow(window_name, img)
                    cv2.waitKey(2000)
                    cv2.destroyWindow(window_name)

    except socket.timeout:
        pass
    except:
        pass