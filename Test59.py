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
image_path = "test_image.png"   # change if needed
image = cv2.imread(image_path)

if image is None:
    raise IOError("Could not load image")

h, w = image.shape[:2]

# ---------------- Windows (Resizable + Positioned) ----------------
cv2.namedWindow("Windscreen Projection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Windscreen Projection", 1280, 720)
cv2.resizeWindow("Calibration", 400, 300)

cv2.moveWindow("Windscreen Projection", 50, 50)
cv2.moveWindow("Calibration", 1400, 50)

# ---------------- Trackbars (Boss-friendly labels) ----------------
cv2.createTrackbar("Top Left", "Calibration", 0, 300, nothing)
cv2.createTrackbar("Top Right", "Calibration", 80, 300, nothing)
cv2.createTrackbar("Bottom Right", "Calibration", 720, 1200, nothing)
cv2.createTrackbar("Bottom Left", "Calibration", 800, 1200, nothing)

# ---------------- Main Loop ----------------
while True:
    # Read slider values
    top_left = cv2.getTrackbarPos("Top Left", "Calibration")
    top_right = cv2.getTrackbarPos("Top Right", "Calibration")
    bottom_right = cv2.getTrackbarPos("Bottom Right", "Calibration")
    bottom_left = cv2.getTrackbarPos("Bottom Left", "Calibration")

    # Destination points (internal logic)
    dst_points = np.float32([
        [0, top_left],          # Top-left
        [w, top_right],         # Top-right
        [w, bottom_right],      # Bottom-right
        [0, bottom_left]        # Bottom-left
    ])

    warped_image = simulate_windscreen_projection(image, dst_points)

    cv2.imshow("Windscreen Projection", warped_image)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
