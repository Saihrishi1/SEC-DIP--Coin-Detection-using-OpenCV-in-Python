# WORKSHOP - 4 : Coin Detection using OpenCV in Python

### Name : Sai Hrishi M
### Reg.No : 212224240140

## Aim :

To develop an AI-based image processing system that can automatically detect and count coins in an image using Python and OpenCV, while visualizing all the intermediate processing steps such as grayscale conversion, blurring, edge detection, and contour detection.

## OBJECTIVE :

1. To apply fundamental computer vision techniques to identify circular objects (coins).

2. To understand the use of image preprocessing and feature extraction using OpenCV.

3. To display all intermediate outputs to explain how detection is achieved.

4. To count and label the number of coins accurately.

## ALGORITHM :

1. Start

2. Input the image (coins image file).

3. Convert the image to grayscale to simplify analysis.

4. Apply Gaussian Blur to reduce image noise and smooth edges.

5. Apply Canny Edge Detection to find edges of coins.

6. Find Contours in the edge-detected image.

7. Filter Contours based on area (to remove small noise).

8.Draw circles around detected coins and assign serial numbers.

9.Count the total number of coins detected.

10. End.

## Program :

```py

import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

img=cv2.imread(r"C:\Users\admin\Downloads\CoinsA.png")

image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

imageCopy = image.copy()
plt.imshow(image)
plt.title("Original Image")
plt.show()

imageGray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(12,12))
plt.subplot(121);plt.imshow(image);plt.title("Original Image")
plt.subplot(122); plt.imshow(imageGray,cmap='gray');plt.title("Grayscale Image");
plt.show()

# Split Image into R,G,B Channels

imageR,imageG,imageB=cv2.split(image)

plt.figure(figsize=(20,12))
plt.subplot(141);plt.imshow(image);plt.title("Original Image")
plt.subplot(142);plt.imshow(imageB,cmap='gray');plt.title("Blue Channel")
plt.subplot(143);plt.imshow(imageG,cmap='gray');plt.title("Green Channel")
plt.subplot(144);plt.imshow(imageR,cmap='gray');plt.title("Red Channel");
plt.show()


# Perform Thresholding

ret, thresh_inv = cv2.threshold(imageG, 20,255, cv2.THRESH_BINARY_INV)

plt.imshow(thresh_inv,cmap='gray');
plt.title("Original Image")
plt.show()


# Perform morphological operations

kernel=np.ones((8,8),dtype=np.uint8)
dilution=cv2.dilate(thresh_inv,kernel,iterations=1)
plt.imshow(dilution,cmap='gray');plt.title('Dilated Image Iteration 2');plt.show

erosion=cv2.erode(dilution,kernel,iterations=1)
plt.imshow(erosion,cmap='gray');plt.title("Eroded Image");plt.show()

# Create SimpleBlobDetector
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.8


detector = cv2.SimpleBlobDetector_create(params)



# Detect Coins


keypoints = detector.detect(erosion)


# Print number of coins detected

print(f"Number of coins detected: {len(keypoints)}")


for k in keypoints:
    x,y =k.pt
    x=int(round(x))
    y=int(round(y))

    cv2.circle(image,(x,y),5,(255,0,0),-1)

    diameter = k.size
    radius = int(round(diameter/2))

    cv2.circle(image,(x,y),radius,(0,255,0),2)

plt.imshow(image,cmap="gray")
plt.title("Fianl Image")
plt.show()


```


# Output :

<img width="950" height="500" alt="download" src="https://github.com/user-attachments/assets/a52fbd80-94e2-4bca-bfc8-3212f0711f5d" />

<img width="1990" height="456" alt="download" src="https://github.com/user-attachments/assets/4eb1b732-9f46-4e32-b1fc-f42d7a07dfd1" />

# Result :

The system successfully detected and counted all coins in the given image.
