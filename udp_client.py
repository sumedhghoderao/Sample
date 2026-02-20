import socket

SERVER_IP = "172.20.10.15"   # Jetson IP
SERVER_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Register with server
sock.sendto(b"HELLO", (SERVER_IP, SERVER_PORT))
print("📨 Sent HELLO to server")

while True:
    data, addr = sock.recvfrom(1024)
    message = data.decode()

    if message == "ACK":
        print("✅ Server acknowledged connection")

    elif message == "FACE_DETECTED":
        print("🚨 FACE DETECTED SIGNAL RECEIVED")
