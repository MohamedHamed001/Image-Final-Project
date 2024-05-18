import cv2
import numpy as np
from collections import defaultdict
import math

def get_edges(img, min_edge_threshold=100, max_edge_threshold=200):
    # Convert to gray scale
    min_edge_threshold = 0.2 * np.max(img)
    max_edge_threshold = 0.6 * np.max(img)

    # Edge detection on the input image
    edge_image = cv2.Canny(img, min_edge_threshold, max_edge_threshold)
    return edge_image

def precompute_trig(theta_range):
    cos_theta = {}
    sin_theta = {}
    for theta in theta_range:
        cos_theta[theta] = np.cos(theta)
        sin_theta[theta] = np.sin(theta)
    return cos_theta, sin_theta

# Load and preprocess the image
image = cv2.imread('assets/test.png')
print("Image loaded.")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Image converted to grayscale.")
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
print("Gaussian blur applied.")
edges = get_edges(blurred)
print("Edge detection done.")

# Parameters
a_range = (10, 20)
b_range = (10, 20)
theta_range = np.deg2rad(np.arange(0, 90))  # Convert theta to radians
cos_theta, sin_theta = precompute_trig(theta_range)
print("Parameters set.")

# Initialize the accumulator as a dictionary
accumulator = defaultdict(int)
print("Accumulator initialized.")

# Voting
for y, x in np.argwhere(edges):
    for a in range(*a_range):
        for b in range(*b_range):
            for theta in theta_range:
                x_c = int(round(x - a * cos_theta[theta]))
                y_c = int(round(y + b * sin_theta[theta]))
                if 0 <= x_c < image.shape[1] and 0 <= y_c < image.shape[0]:
                    accumulator[(y_c, x_c, a, b, theta)] += 1
print("Voting completed.")

# Find potential ellipses
max_accumulator = max(accumulator.values())
threshold = max_accumulator * 0.5
potential_ellipses = [k for k, v in accumulator.items() if v >= threshold]
print(f"Potential ellipses identified: {len(potential_ellipses)}")

# Ellipse Fitting
scale_factor = 0.1  # Further reduce the size of the ellipse
for y_c, x_c, a, b, theta in potential_ellipses:
    scaled_a = int(a * scale_factor)
    scaled_b = int(b * scale_factor)
    cv2.ellipse(image, (x_c, y_c), (scaled_a, scaled_b), np.degrees(theta), 0, 360, (0, 255, 0), 1)
print("Ellipses drawn on image.")

# Display Results
cv2.imshow('Detected Ellipses', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Finished displaying results.")
