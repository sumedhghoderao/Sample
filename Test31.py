import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Load the pre-trained DNN model for face detection (Caffe model)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Function to simulate an image placed on a curved surface (projection onto windscreen)
def simulate_windscreen_projection(image, dst_points):
    h, w = image.shape[:2]
    src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])  # Corners of the input image
    M = cv2.getPerspectiveTransform(src_points, dst_points)  # Perspective transformation matrix
    warped_image = cv2.warpPerspective(image, M, (w, h))  # Apply the perspective warp
    return warped_image

# Function to initialize the camera (Picamera2)
def initialize_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
    picam2.start()
    return picam2

# Function to detect faces using the DNN model
def detect_faces(frame):
    # Prepare the frame for the DNN face detector
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold (can be adjusted)
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# Function to simulate the windscreen effect on a frame
def simulate_effect_on_frame(frame, offset_x):
    # Hardcoded base coordinates for the windscreen corners
    x1, y1 = 0, 0  # Top-left corner
    x2, y2 = 640, 0  # Top-right corner
    x3, y3 = 640, 480  # Bottom-right corner
    x4, y4 = 0, 480  # Bottom-left corner

    # Adjust x1 and x2 for the windscreen effect
    x1 += offset_x      # Move x1 to the right
    x2 -= offset_x      # Move x2 to the left (opposite of x1)

    dst_points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    warped_frame = simulate_windscreen_projection(frame, dst_points)
    
    return warped_frame

# Main function
def main():
    picam2 = initialize_camera()

    prev_offset_x = 0  # To store the previous offset
    frame_counter = 0  # Frame counter to manage the FPS
    last_face_detection_time = time.time()  # For face detection timing
    offset_x = 0  # Initial offset (will be updated by face detection)

    while True:
        frame = picam2.capture_array()
        if frame is None:
            print("Failed to capture frame")
            break
        
        # If it's the first frame or every 30th frame, perform face detection
        if frame_counter == 0:
            faces = detect_faces(frame)
            print(f"Detected faces: {faces}")  # Debug information
            
            for (x, y, w, h) in faces:
                # Calculate the offset based on the face's position (simple adjustment)
                eye_center_y = y + 20  # Approximate center of the face
                offset_x = -(eye_center_y - frame.shape[0] // 2) * 0.1  # Inverted multiplier for effect
                
                # Store the offset for future frames
                prev_offset_x = offset_x

        # Apply the same perspective transformation to all subsequent frames
        warped_frame = simulate_effect_on_frame(frame, prev_offset_x)

        # Display the warped frame
        cv2.imshow("Curved Surface Simulation (Windscreen-like)", warped_frame)

        # Increment the frame counter
        frame_counter += 1
        if frame_counter >= 30:
            frame_counter = 0  # Reset the frame counter every 30 frames

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
