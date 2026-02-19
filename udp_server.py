import socket
import time

UDP_IP = "0.0.0.0"   # listen on all interfaces
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("🟢 UDP Server started on port 9999")
print("Waiting for client registration...")

client_addr = None
last_sent = 0

while True:
    try:
        # Non-blocking receive (timeout)
        sock.settimeout(0.5)

        data, addr = sock.recvfrom(1024)
        message = data.decode()

        if message == "HELLO":
            client_addr = addr
            print(f"✅ Client registered: {client_addr}")
            sock.sendto(b"ACK", client_addr)

    except socket.timeout:
        pass

    # ---- Simulate FACE DETECTED every 5 seconds ----
    if client_addr and time.time() - last_sent > 5:
        sock.sendto(b"FACE_DETECTED", client_addr)
        print("📡 Sent: FACE_DETECTED")
        last_sent = time.time()
