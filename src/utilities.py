import numpy as np

# Get index and value of image
def getCenter(image):
    xCenter = image.shape[0] // 2
    yCenter = image.shape[1] // 2
    centerCoordinates = (xCenter,yCenter)
    return centerCoordinates, image[centerCoordinates]
