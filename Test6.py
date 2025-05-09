import cv2
import numpy as np

# Load DNN face detector model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Detect faces using DNN
def detect_face_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))

    return faces

# Perspective warp function
def simulate_windscreen_projection(image, dst_points):
    h, w = image.shape[:2]
    src_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, M, (w, h))
    return warped_image, M

# Barrel distortion function
def apply_barrel_distortion(image, k1=0.5, k2=0.0, p1=0.001, p2=0.001, k3=0.0):
    h, w = image.shape[:2]
    fx = fy = w
    cx = w / 2
    cy = h / 2
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
    )
    distorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return distorted

# Open webcam
video_input = cv2.VideoCapture(0)

# Open input video
image_video = cv2.VideoCapture("input_video.mp4")

# Verify webcam and video loaded
if not video_input.isOpened() or not image_video.isOpened():
    print("Error: Could not open webcam or input video.")
else:
    # Set webcam to 1280x720
    video_input.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_input.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoothing_factor = 0.1
    prev_offset_x = 0
    frame_count = 0
    frame_skip = 1

    MIN_OFFSET_X = -100
    MAX_OFFSET_X = 100

    while True:
        ret_webcam, frame_webcam = video_input.read()
        ret_video, image_input = image_video.read()

        # Loop video if it ends
        if not ret_video:
            image_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if not ret_webcam:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame_webcam = cv2.resize(frame_webcam, (1280, 720))
        image_input = cv2.resize(image_input, (1280, 720))
        faces = detect_face_dnn(frame_webcam)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame_webcam, (x, y), (x + w, y + h), (255, 0, 0), 2)

            eye_center_y = y + h // 2
            raw_offset_x = (eye_center_y - frame_webcam.shape[0] // 2) * 0.4

            # Smooth and clamp horizontal offset
            offset_x = prev_offset_x + smoothing_factor * (raw_offset_x - prev_offset_x)
            offset_x = np.clip(offset_x, MIN_OFFSET_X, MAX_OFFSET_X)
            prev_offset_x = offset_x

            # Base corners
            x1, y1 = 0, 0
            x2, y2 = 1280, 0
            x3, y3 = 1280, 720
            x4, y4 = 0, 720

            # Apply x offsets
            x1 += offset_x + 30
            x2 += offset_x + 40
            x3 -= offset_x + 60
            x4 -= offset_x

            # Apply y offsets
            offset_y1 = 0
            offset_y2 = 20
            offset_y3 = -50
            offset_y4 = -30
            y1 += offset_y1
            y2 += offset_y2
            y3 += offset_y3
            y4 += offset_y4

            dst_points = np.float32([
                [x1, y1], [x2, y2], [x3, y3], [x4, y4]
            ])

            # Apply perspective warp
            warped_webcam, _ = simulate_windscreen_projection(frame_webcam, dst_points)
            warped_image, _ = simulate_windscreen_projection(image_input, dst_points)

            # Apply intensity correction only to the warped image
            height, width = warped_image.shape[:2]
            gradient = np.linspace(1.0, 0.5, width)  # Dimming from left to right
            correction_mask = np.tile(gradient, (height, 1))
            correction_mask = np.stack([correction_mask] * 3, axis=2).astype(np.float32)

            # Normalize, apply mask, and convert back
            normalized_image = warped_image.astype(np.float32) / 255.0
            corrected_image = normalized_image * correction_mask
            corrected_image = np.clip(corrected_image, 0, 1)
            corrected_image = (corrected_image * 255).astype(np.uint8)

            # Apply barrel distortion
            distorted_webcam = apply_barrel_distortion(warped_webcam)
            distorted_image = apply_barrel_distortion(corrected_image)

            # Show output
            cv2.imshow("Webcam with Windscreen and Barrel Effect", distorted_webcam)
            cv2.imshow("Image with Windscreen and Barrel Effect", distorted_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_input.release()
    image_video.release()
    cv2.destroyAllWindows()
