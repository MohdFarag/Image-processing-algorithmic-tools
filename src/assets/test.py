from math import *
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Get index and value of image
def getCenter(image:np.ndarray):
    return (image.shape[0] // 2,image.shape[1] // 2), image[(image.shape[0] // 2,image.shape[1] // 2)]

def naive_image_rotate(image, degrees, option='same'):
    '''
    This function rotates the image around its center by amount of degrees
    provided. The rotated image can be of the same size as the original image
    or it can show the full image.
    
    inputs: image: input image (dtype: numpy-ndarray)
            degrees: amount of rotation in degrees (e.g., 45,90 etc.)
            option: string variable for type of rotation. It can take two values
            'same': the rotated image will have same size as the original image
                    It is default value for this variable.
            'full': the rotated image will show the full rotation of original
                    image thus the size may be different than original.
    '''
    # First we will convert the degrees into radians
    rads = radians(degrees)
    # Finding the center point of the original image
    cx, cy = (image.shape[1]//2, image.shape[0]//2)
    
    if(option!='same'):
        # Let us find the height and width of the rotated image
        height_rot_img = round(abs(image.shape[0]*sin(rads))) + \
                           round(abs(image.shape[1]*cos(rads)))
        width_rot_img = round(abs(image.shape[1]*cos(rads))) + \
                           round(abs(image.shape[0]*sin(rads)))
        rot_img = np.uint8(np.zeros((height_rot_img,width_rot_img,image.shape[2])))
        # Finding the center point of rotated image.
        midx,midy = (width_rot_img//2, height_rot_img//2)
    else:
        rot_img = np.uint8(np.zeros(image.shape))
     
    for i in range(rot_img.shape[0]):
        for j in range(rot_img.shape[1]):
            if(option!='same'):
                x= (i-midx)*cos(rads)+(j-midy)*sin(rads)
                y= -(i-midx)*sin(rads)+(j-midy)*cos(rads)
                x=round(x)+cy
                y=round(y)+cx
            else:
                x= (i-cx)*cos(rads)+(j-cy)*sin(rads)
                y= -(i-cx)*sin(rads)+(j-cy)*cos(rads)
                x=round(x)+cx
                y=round(y)+cy

            if (x>=0 and y>=0 and x<image.shape[0] and  y<image.shape[1]):
                rot_img[i,j] = image[x,y]
    return rot_img

# Rotate image
def rotate(image, angle, mode="nearest"):
    # Converting degrees to radians
    angle = radians(-angle)
    
    # Cosine & Sine
    cosine = cos(angle)
    sine = sin(angle)

    # Define the width of the image
    oldWidth = image.shape[0]
    # Define the height of the image
    oldHeight = image.shape[1]
    
    # Initialize rotated image 
    rotatedImage = np.zeros((oldWidth,oldHeight)) 

    # Find the center of the rotated image
    centerWidth, centerHeight = (oldWidth//2, oldHeight//2)

    if mode == "nearest":
        for i in range(oldWidth):
            for j in range(oldHeight):
                x = -(j-centerHeight)*sine + (i-centerWidth)*cosine
                y = (j-centerHeight)*cosine + (i-centerWidth)*sine

                # Add offset
                x += centerWidth
                y += centerHeight
                
                # Get nearest index
                x, y = round(x), round(y) 

                # Check if x/y corresponds to a valid pixel in input image
                if (0 <= x < oldWidth and  0 <= y < oldHeight):
                    rotatedImage[i][j] = image[x][y]

    elif mode == "linear":    
        for i in range(oldWidth):
            for j in range(oldHeight):
                x = -(j-centerHeight)*sine + (i-centerWidth)*cosine
                y = (j-centerHeight)*cosine + (i-centerWidth)*sine

                # Add offset
                x += centerWidth
                y += centerHeight
                
                # Calculate the coordinate values for 4 surrounding pixels.
                x_floor = int(x)
                x_ceil = min(oldWidth-1, ceil(x))
                y_floor = int(y)
                y_ceil = min(oldHeight - 1, ceil(y))
                
                if (0 <= x < oldWidth and  0 <= y < oldHeight):
                    if (x_ceil == x_floor) and (y_ceil == y_floor):
                        q = image[int(x), int(y)]
                    elif (y_ceil == y_floor):
                        q1 = image[x_floor, int(y)]
                        q2 = image[x_ceil, int(y)]
                        q = q1 * (x_ceil - x) + q2 * (x - x_floor)
                    elif (x_ceil == x_floor):
                        q1 = image[int(x), y_floor]
                        q2 = image[int(x), y_ceil]
                        q = (q1 * (y_ceil - y)) + (q2 * (y - y_floor))
                    else:
                        p1 = image[x_floor, y_floor]
                        p2 = image[x_floor, y_ceil]
                        p3 = image[x_ceil, y_floor]
                        p4 = image[x_ceil, y_ceil]

                        q1 = p1 * (y_ceil - y) + p2 * (y - y_floor)
                        q2 = p3 * (y_ceil - y) + p4 * (y - y_floor)
                        q = q1 * (x_ceil - x) + q2 * (x - x_floor)
                    rotatedImage[i][j] = q

    return rotatedImage


image = mpimg.imread('./testInputs/Camera man/Original-cameraman-image_Q640.jpg')
plt.imshow(image, cmap="gray")
plt.show()

start = time.time()
fastImage = naive_image_rotate(image,45)
end = time.time()
timeFastRotate = end - start
print(timeFastRotate)

plt.imshow(fastImage, cmap="gray")
plt.show()

#############

start = time.time()
slowImage = rotate(image,45)
end = time.time()
timeNormalRotate = end - start
print(timeNormalRotate)


plt.imshow(slowImage, cmap="gray")
plt.show()

##

if timeFastRotate<timeNormalRotate:
    print("Fast rotate is faster than normal rotate")
else:
    print("Normal rotate is faster than fast rotate")

# a = np.arange(25).reshape((5,5))
# print(a)
# print(a[np.ix_([1,3,0], [0,2,0,4])])