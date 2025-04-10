import cv2
import numpy as np

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

# Open the webcam feed
video_input = cv2.VideoCapture(0)

# Open the input video file
video_file = cv2.VideoCapture("input_video.mp4")

if not video_input.isOpened() or not video_file.isOpened():
    print("Error: Could not open video.")
else:
    # Set the resolution to lower for better performance
    video_input.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640
    video_input.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480

    # Initialize smoothing factor and previous offset
    smoothing_factor = 0.1
    prev_offset_x = 0
    frame_skip = 1  # Skip every 8th frame to reduce processing load
    frame_count = 0  # Counter for skipping frames
    
    # Define the offset limits
    MIN_OFFSET_X = -30  # Minimum shift value
    MAX_OFFSET_X = 30   # Maximum shift value

    while True:
        ret1, frame_webcam = video_input.read()
        ret2, frame_video = video_file.read()
        
        if not ret1 or not ret2:
            # Restart the video file from the beginning if it's at the end
            video_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue  # Skip the current frame and continue looping

        frame_count += 1

        # Skip frames to optimize performance (frame skipping)
        if frame_count % frame_skip != 0:
            continue

        # Detect faces in the webcam frame using DNN
        faces = detect_face_dnn(frame_webcam)
        
        for (x, y, w, h) in faces:
            # Draw a square around the face in the webcam feed
            cv2.rectangle(frame_webcam, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Calculate the vertical center of the face box to adjust the windscreen effect
            eye_center_y = y + h // 2

            # Adjust the corner coordinates based on the vertical position of the face
            offset_x = (eye_center_y - frame_webcam.shape[0] // 2) * 1  # Adjust multiplier for effect

            # Apply smoothing to prevent sudden jumps in offset
            offset_x = prev_offset_x + smoothing_factor * (offset_x - prev_offset_x)

            # Clamp the offset_x to be within the specified limits
            offset_x = np.clip(offset_x, MIN_OFFSET_X, MAX_OFFSET_X)

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
            warped_frame_webcam, M = simulate_windscreen_projection(frame_webcam, dst_points)

            # Apply the same transformation (offset correction) to the video frame
            warped_frame_video, _ = simulate_windscreen_projection(frame_video, dst_points)

            # Show the warped frame in separate windows for webcam and video
            cv2.imshow("Webcam with Windscreen Effect", warped_frame_webcam)
            cv2.imshow("Video with Windscreen Effect", warped_frame_video)

        # Press 'q' to exit the video preview
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_input.release()
    video_file.release()
    cv2.destroyAllWindows()
