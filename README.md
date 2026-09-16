# Digital Image Processing Techniques.
# Exp : Workshop-1
# Name : JISHNUPRIYAN S
# Reg. No : 212223240061

## AIM :
Our aim is to adding Sunglasses to Your Passport Photo Using OpenCV.


## Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look
1. Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

2. Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
1. Detects the face in an image.
2. Places a stylish sunglass overlay perfectly on the face.
3. Works seamlessly with individual passport-size photos.
4. Customizable for different sunglasses styles or photo types.

## Technologies Used:
1. Python
2. OpenCV for image processing
3. Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the images folder.
3. Run the script to see your "cool" transformation!

## Applications:
1. Learning basic image processing techniques.
2. Adding flair to your photos for fun.
3. Practicing computer vision workflows.


## PROGRAM :

```py

import cv2
import numpy as np
import matplotlib.pyplot as plt
faceImage = cv2.imread("sank.jpeg")

print("Face image shape:", faceImage.shape)

plt.figure(figsize=(8, 10))
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Original Face")
plt.axis("off")
plt.show()
glassPNG = cv2.imread("gl.jpg", cv2.IMREAD_UNCHANGED)

print("Original sunglasses shape:", glassPNG.shape)
if glassPNG.shape[2] == 4:

    # PNG has an alpha channel
    glassBGR = glassPNG[:, :, :3]
    glassMask1 = glassPNG[:, :, 3]

else:

    # PNG has no alpha channel
    print("WARNING: gl.jpg does not have an alpha channel.")

    glassBGR = glassPNG

    # Create a mask from non-black pixels
    gray = cv2.cvtColor(glassBGR, cv2.COLOR_BGR2GRAY)
    _, glassMask1 = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)


# Display sunglasses and mask

plt.figure(figsize=(15, 6))

plt.subplot(121)
plt.imshow(cv2.cvtColor(glassBGR, cv2.COLOR_BGR2RGB))
plt.title("Sunglasses")
plt.axis("off")

plt.subplot(122)
plt.imshow(glassMask1, cmap="gray")
plt.title("Sunglasses Mask")
plt.axis("off")

plt.show()
# Find pixels belonging to the sunglasses
ys, xs = np.where(glassMask1 > 0)

if len(xs) == 0:
    raise ValueError("The sunglasses mask is empty.")

# Bounding box
x_min = xs.min()
x_max = xs.max()
y_min = ys.min()
y_max = ys.max()

print("Sunglasses bounding box:")
print("x:", x_min, "to", x_max)
print("y:", y_min, "to", y_max)


# Crop both image and mask
glassBGR = glassBGR[y_min:y_max+1, x_min:x_max+1]
glassMask1 = glassMask1[y_min:y_max+1, x_min:x_max+1]

print("Cropped sunglasses:", glassBGR.shape)
faceWithGlassesNaive = faceImage.copy()

x1 = 300
y1 = 360

# Width of the sunglasses on the face
new_width = 620


original_height, original_width = glassBGR.shape[:2]

new_height = int(
    original_height * new_width / original_width
)

glassBGR_resized = cv2.resize(
    glassBGR,
    (new_width, new_height),
    interpolation=cv2.INTER_AREA
)

glassMask1_resized = cv2.resize(
    glassMask1,
    (new_width, new_height),
    interpolation=cv2.INTER_NEAREST
)

print("Resized sunglasses:", glassBGR_resized.shape)

face_height, face_width = faceImage.shape[:2]

if x1 + new_width > face_width:
    raise ValueError("Sunglasses extend beyond the right side of the face image.")

if y1 + new_height > face_height:
    raise ValueError("Sunglasses extend beyond the bottom of the face image.")

eyeROI = faceWithGlassesNaive[
    y1:y1+new_height,
    x1:x1+new_width
]

print("Face ROI:", eyeROI.shape)
print("Glass:", glassBGR_resized.shape)

glassMask = cv2.merge(
    (
        glassMask1_resized,
        glassMask1_resized,
        glassMask1_resized
    )
)

glassMask = glassMask.astype(np.float32) / 255.0


maskedEye = cv2.multiply(
    eyeROI.astype(np.float32),
    1 - glassMask
)



maskedGlass = cv2.multiply(
    glassBGR_resized.astype(np.float32),
    glassMask
)

eyeRoiFinal = cv2.add(
    maskedEye,
    maskedGlass
)

eyeRoiFinal = np.uint8(eyeRoiFinal)


faceWithGlassesNaive[
    y1:y1+new_height,
    x1:x1+new_width
] = eyeRoiFinal

plt.figure(figsize=(20, 6))

plt.subplot(131)
plt.imshow(cv2.cvtColor(np.uint8(maskedEye), cv2.COLOR_BGR2RGB))
plt.title("Masked Face")
plt.axis("off")

plt.subplot(132)
plt.imshow(cv2.cvtColor(np.uint8(maskedGlass), cv2.COLOR_BGR2RGB))
plt.title("Masked Sunglasses")
plt.axis("off")

plt.subplot(133)
plt.imshow(cv2.cvtColor(eyeRoiFinal, cv2.COLOR_BGR2RGB))
plt.title("Combined ROI")
plt.axis("off")

plt.show()

plt.figure(figsize=(16, 8))

plt.subplot(121)
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(122)
plt.imshow(cv2.cvtColor(faceWithGlassesNaive, cv2.COLOR_BGR2RGB))
plt.title("With Sunglasses")
plt.axis("off")

plt.show()


```



## OUTPUT :

### Original Image :
<img width="622" height="796" alt="Screenshot 2026-09-16 234257" src="https://github.com/user-attachments/assets/c91bb71f-8a9e-4dd7-88f3-2ee956cc8b8f" />


### Sunglass :
<img width="1080" height="350" alt="Screenshot 2026-09-16 234308" src="https://github.com/user-attachments/assets/eb280716-880e-4095-88c4-70fb9cbf480a" />


### Boundary Box :
<img width="271" height="90" alt="Screenshot 2026-09-16 234630" src="https://github.com/user-attachments/assets/d00603e7-aaa5-4b6c-80c6-f519067bd1d1" />


### Masked Sunglass Image :
<img width="1110" height="212" alt="Screenshot 2026-09-16 234320" src="https://github.com/user-attachments/assets/2c03fc02-a19a-4729-b47a-818d930e8e6b" />



### Original Image vs Image with Sunglass:
<img width="517" height="602" alt="Screenshot 2026-09-16 234337" src="https://github.com/user-attachments/assets/a01eef00-f867-4a12-b617-a8c594c13b0a" />
<img width="961" height="877" alt="screen shot" src="https://github.com/user-attachments/assets/db3a1c7c-90a0-4a8b-84b2-8b4b90f8480c" />



## RESULT :
Thus, We successfully added Sunglasses to Your Passport Photo Using OpenCV.
