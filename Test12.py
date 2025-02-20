import cv2
import numpy as np

# Function to simulate an image placed on a curved surface (pre-warp the image to compensate for curvature)
def simulate_windscreen_projection(image, dst_points):
    # Get image dimensions
    h, w = image.shape[:2]

    # Define the source points (image corners)
    src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])

    # Calculate the perspective transformation matrix (simulate flat projection)
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective warp (pre-warp the image)
    warped_image = cv2.warpPerspective(image, M, (w, h))

    return warped_image, M

# Function to correct the curvature and unwrap the image (apply reverse distortion)
def correct_projection(image, M):
    # Get image dimensions
    h, w = image.shape[:2]

    # Inverse the perspective matrix to undo the distortion (flatten the image)
    M_inv = np.linalg.inv(M)

    # Apply the inverse perspective transformation to unwarp the image (simulate flat projection)
    corrected_image = cv2.warpPerspective(image, M_inv, (w, h))

    return corrected_image

# Load the image
image = cv2.imread('flat_image.jpg')  # Replace with your image path

# Define destination points to simulate the curved surface (windscreen effect)
h, w = image.shape[:2]
dst_points = np.float32([
    [int(0.1 * w), 0],  # Top-left corner moves inward (simulate the curvature)
    [int(0.9 * w), 0],  # Top-right corner moves inward
    [w-1, h-1],         # Bottom-right stays in place
    [0, h-1]            # Bottom-left stays in place
])

# Simulate the image projected onto the curved surface (windscreen effect)
warped_image, M = simulate_windscreen_projection(image, dst_points)

# Correct the image to remove the curvature (unwrap it)
corrected_image = correct_projection(warped_image, M)

# Show the results
cv2.imshow("Curved Surface Simulation (Windscreen-like)", warped_image)
cv2.imshow("Corrected Image", corrected_image)

# Wait until the user presses a key
cv2.waitKey(0)

# Cleanup
cv2.destroyAllWindows()
