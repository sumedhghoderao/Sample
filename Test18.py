import cv2
import numpy as np

# Function to simulate an image placed on a curved surface (projection onto windscreen)
def simulate_windscreen_projection(image, dst_points):
    # Get image dimensions
    h, w = image.shape[:2]

    # Define the source points (image corners)
    src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])

    # Calculate the perspective transformation matrix (simulate projection onto windscreen)
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective warp (pre-warp the image)
    warped_image = cv2.warpPerspective(image, M, (w, h))

    return warped_image, M

# Open the webcam feed (0 for default webcam)
video_input = cv2.VideoCapture(0)

if not video_input.isOpened():
    print("Error: Could not open webcam.")
else:
    # Set the resolution to a lower value (for performance)
    video_input.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the width to 1280
    video_input.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set the height to 720

    # Load the face and eye cascades for detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Get video properties
    fps = video_input.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer (optional: save webcam feed as a video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_output = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

    # Initialize the smoothing factor for x1 and x2 adjustments
    smoothing_factor = 0.1  # The lower the value, the smoother the transition
    prev_offset_x = 0

    frame_skip = 8  # Skip every 2 frames to reduce processing load
    frame_count = 0  # Counter for skipping frames

    while True:
        # Read the next frame from the webcam feed
        ret, frame = video_input.read()

        if not ret:
            break  # Exit the loop if there are no more frames

        frame_count += 1

        # Skip frames to optimize performance
        if frame_count % frame_skip != 0:
            continue

        # Convert the frame to grayscale for face and eye detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Detect eyes within the face region
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) > 0:
                # We take the first detected eye (or you can adjust if multiple eyes are detected)
                ex, ey, ew, eh = eyes[0]

                # Calculate the vertical center of the eye box
                eye_center_y = ey + eh // 2

                # Adjust the hardcoded corner coordinates based on the vertical position of the eye
                offset_x = (eye_center_y - h // 2) * 3  # Adjust this multiplier for more/less effect

                # Apply smoothing to prevent sudden jumps
                offset_x = prev_offset_x + smoothing_factor * (offset_x - prev_offset_x)

                # Update the previous offset value for the next frame
                prev_offset_x = offset_x

                # Hardcoded base coordinates for the windscreen corners
                x1, y1 = 0, 0  # Top-left corner
                x2, y2 = 1280, 0  # Top-right corner
                x3, y3 = 1280, 720  # Bottom-right corner
                x4, y4 = 0, 720   # Bottom-left corner

                # Adjust the corners based on eye movement (apply vertical adjustment)
                x1 += offset_x
                x2 += offset_x

                # Define the destination points (the corners that will be moved as per the adjusted values)
                dst_points = np.float32([ 
                    [x1, y1],  # Top-left corner
                    [x2, y2],  # Top-right corner
                    [x3, y3],  # Bottom-right corner
                    [x4, y4]   # Bottom-left corner
                ])

                # Simulate the image projected onto the curved surface (windscreen effect)
                warped_frame, M = simulate_windscreen_projection(frame, dst_points)

                # Write the warped frame to the output video (optional)
                video_output.write(warped_frame)

                # Optionally, show the results (you can disable this to just save the video)
                cv2.imshow("Curved Surface Simulation (Windscreen-like)", warped_frame)

        # Press 'q' to exit the video preview
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_input.release()
    video_output.release()
    cv2.destroyAllWindows()
