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

# Load the image (assuming it's in the same folder as the script)
image = cv2.imread('flat_image.jpg')

if image is None:
    print("Error: Could not load the image. Please make sure 'flat_image.jpg' is in the same folder.")
else:
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Hardcoded corner coordinates for simulation
    x1, y1 = 100, 50  # Top-left corner
    x2, y2 = 500, 50  # Top-right corner
    x3, y3 = 550, 400  # Bottom-right corner
    x4, y4 = 50, 400   # Bottom-left corner

    # Define the destination points (the corners that will be moved as per the hardcoded values)
    dst_points = np.float32([
        [x1, y1],  # Top-left corner
        [x2, y2],  # Top-right corner
        [x3, y3],  # Bottom-right corner
        [x4, y4]   # Bottom-left corner
    ])

    # Simulate the image projected onto the curved surface (windscreen effect)
    warped_image, M = simulate_windscreen_projection(image, dst_points)

    # Show the results
    cv2.imshow("Curved Surface Simulation (Windscreen-like)", warped_image)

    # Wait until the user presses a key
    cv2.waitKey(0)

    # Cleanup
    cv2.destroyAllWindows()
