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
    # Set the resolution to 2K (1920x1080)
    video_input.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the width
    video_input.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set the height

    # Get video properties
    fps = video_input.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer (optional: save webcam feed as a video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    video_output = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

    while True:
        # Read the next frame from the webcam feed
        ret, frame = video_input.read()

        if not ret:
            break  # Exit the loop if there are no more frames

        # Hardcoded corner coordinates for simulation (same as in your previous example)
        x1, y1 = 80, 0  # Top-left corner
        x2, y2 = 1200, 0  # Top-right corner
        x3, y3 = 1280, 720  # Bottom-right corner
        x4, y4 = 0, 720   # Bottom-left corner

        # Define the destination points (the corners that will be moved as per the hardcoded values)
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
