import cv2
import numpy as np

# ---------------- Utility ----------------
def nothing(x):
    pass

def simulate_windscreen_projection(image, dst_points):
    h, w = image.shape[:2]
    src_points = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

# ---------------- Load Image ----------------
image_path = "test_image.png"  # change if needed
image = cv2.imread(image_path)

if image is None:
    raise IOError("Could not load image")

h, w = image.shape[:2]

# ---------------- Windows ----------------
cv2.namedWindow("Windscreen Projection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

# Move calibration window
cv2.resizeWindow("Calibration", 400, 300)
cv2.moveWindow("Calibration", 50, 50)

# Start fullscreen
fullscreen = True
cv2.setWindowProperty(
    "Windscreen Projection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ---------------- Trackbars ----------------
cv2.createTrackbar("Top Left", "Calibration", 0, 300, nothing)
cv2.createTrackbar("Top Right", "Calibration", 80, 300, nothing)
cv2.createTrackbar("Bottom Right", "Calibration", 720, 1200, nothing)
cv2.createTrackbar("Bottom Left", "Calibration", 800, 1200, nothing)

# ---------------- Main Loop ----------------
while True:
    # Read trackbar values
    top_left = cv2.getTrackbarPos("Top Left", "Calibration")
    top_right = cv2.getTrackbarPos("Top Right", "Calibration")
    bottom_right = cv2.getTrackbarPos("Bottom Right", "Calibration")
    bottom_left = cv2.getTrackbarPos("Bottom Left", "Calibration")

    # Destination points
    dst_points = np.float32([
        [0, top_left],
        [w, top_right],
        [w, bottom_right],
        [0, bottom_left]
    ])

    # Warp image
    warped_image = simulate_windscreen_projection(image, dst_points)

    # Display
    cv2.imshow("Windscreen Projection", warped_image)

    key = cv2.waitKey(1) & 0xFF

    # ESC to exit
    if key == 27:
        break

    # Press 'F' to toggle fullscreen
    if key == ord('f') or key == ord('F'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty(
                "Windscreen Projection",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.setWindowProperty(
                "Windscreen Projection",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )

cv2.destroyAllWindows()
