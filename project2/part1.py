# Here is the implementation with OpenCv, your project will use your own functions. 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import image
img = cv2.imread('./face.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Define the Sobel kernel for horizontal and vertical edges
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

# Apply filter2D to get the horizontal and vertical edges
sobelx = cv2.filter2D(blur, cv2.CV_64F, sobel_kernel_x)
sobely = cv2.filter2D(blur, cv2.CV_64F, sobel_kernel_y)
# Make sure that sobelx and sobely have the same size and type
# sobelx = sobelx.astype(np.float64)
# sobely = sobely.astype(np.float64)

# Compute the magnitude of the gradients
mag = cv2.magnitude(sobelx, sobely)

# Apply thresholding to get a binary image.  Returns two values, 
# the threshold value and the thresholded image.
val, thresh = cv2.threshold(mag, 120, 255, cv2.THRESH_BINARY)

# Display the original image and the resulting edge map using Matplotlib
# Plot size
fig = plt.figure(figsize = (15,15))

# Create subplots for each image
# Left Image
fig.add_subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

# Right Image
fig.add_subplot(1, 2, 2)
plt.imshow(thresh, cmap='gray')
plt.title('Filter2D Edge Detection')