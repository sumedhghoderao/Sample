import cv2
import numpy as np
from picamera2 import Picamera2, Preview

# Load DNN face detector model (download the model files from OpenCV repository)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Function to detect faces using DNN
def detect_face_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faces = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Only keep detections with high confidence (e.g., >50%)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    
    return faces

# Function to simulate an image placed on a curved surface (projection onto windscreen)
def simulate_windscreen_projection(image, dst_points):
    h, w = image.shape[:2]
    src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])  # Corners of the input image
    M = cv2.getPerspectiveTransform(src_points, dst_points)  # Perspective transformation matrix
    warped_image = cv2.warpPerspective(image, M, (w, h))  # Apply the perspective warp
    return warped_image, M

# Initialize the Picamera2 instance
picam2 = Picamera2()

# Configure the camera for preview and video capture
picam2.start_preview(Preview.QTGL)
picam2.start()

# Set the resolution to 640x480 for better performance
picam2.configure({"width": 640, "height": 480})

# Initialize smoothing factor and previous offset
smoothing_factor = 0.1
prev_offset_x = 0
frame_skip = 8  # Skip every 8th frame to reduce processing load
frame_count = 0  # Counter for skipping frames

while True:
    frame = picam2.capture_array()  # Capture the frame from the camera

    if frame is None:
        break  # Exit if no frame is available

    frame_count += 1

    # Skip frames to optimize performance (frame skipping)
    if frame_count % frame_skip != 0:
        continue

    # Detect faces in the frame using DNN
    faces = detect_face_dnn(frame)
    
    for (x, y, w, h) in faces:
        # Draw a square around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Calculate the vertical center of the face box to adjust the windscreen effect
        eye_center_y = y + h // 2

        # Adjust the corner coordinates based on the vertical position of the face
        offset_x = (eye_center_y - frame.shape[0] // 2) * 1  # Adjust multiplier for effect

        # Apply smoothing to prevent sudden jumps in offset
        offset_x = prev_offset_x + smoothing_factor * (offset_x - prev_offset_x)

        # Update previous offset for the next frame
        prev_offset_x = offset_x

        # Hardcoded base coordinates for the windscreen corners
        x1, y1 = 0, 0  # Top-left corner
        x2, y2 = 640, 0  # Top-right corner
        x3, y3 = 640, 480  # Bottom-right corner
        x4, y4 = 0, 480  # Bottom-left corner

        # Synchronized but opposite movement of x1 and x2
        x1 += offset_x      # Move x1 to the right
        x2 -= offset_x      # Move x2 to the left (opposite of x1)

        # Define the destination points after adjustment
        dst_points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        # Simulate the image projected onto the curved surface (windscreen effect)
        warped_frame, M = simulate_windscreen_projection(frame, dst_points)

        # Show the warped frame in the output window
        cv2.imshow("Curved Surface Simulation (Windscreen-like)", warped_frame)

    # Press 'q' to exit the video preview
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
picam2.stop()
cv2.destroyAllWindows()
