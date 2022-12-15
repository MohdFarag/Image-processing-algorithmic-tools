import numpy as np
from math import *


def get_padding_width_per_side(kernel_size: int) -> int:
    # Simple integer division
    return kernel_size // 2
    
def add_padding_to_image(img: np.array, padding_width: int) -> np.array:
    # Array of zeros of shape (img + padding_width)
    img_with_padding = np.zeros(shape=(
        img.shape[0] + padding_width * 2,  # Multiply with two because we need padding on all sides
        img.shape[1] + padding_width * 2
    ))
    
    # Change the inner elements
    # For example, if img.shape = (224, 224), and img_with_padding.shape = (226, 226)
    # keep the pixel wide padding on all sides, but change the other values to be the same as img
    img_with_padding[padding_width:-padding_width, padding_width:-padding_width] = img
    
    return img_with_padding

def calculate_target_size(img_size: int, kernel_size: int) -> int:
    num_pixels = 0
    
    # From 0 up to img size (if img size = 224, then up to 223)
    for i in range(img_size):
        # Add the kernel size (let's say 3) to the current i
        added = i + kernel_size
        # It must be lower than the image size
        if added <= img_size:
            # Increment if so
            num_pixels += 1
            
    return num_pixels

# Get depth of image
def getDepth(image:np.ndarray):
    rangeOfImage = image.max() - image.min()
    bitDepthForOneChannel = ceil(log2(rangeOfImage))
    
    if image.ndim == 3:
        _ , _ , numOfChannels = image.shape
    else:
        numOfChannels = 1

    bitDepth = bitDepthForOneChannel * numOfChannels
    return bitDepth

# Get index and value of image
def getCenter(image):
    xCenter = image.shape[0] // 2
    yCenter = image.shape[1] // 2
    centerCoordinates = (xCenter,yCenter)
    return centerCoordinates, image[centerCoordinates]

# Statistics
def getMean(image):
    return np.mean(image)

def getMedian(image):
    return np.median(image)

def getVariance(image):
    return np.var(image)

# Log transformation
def logTransformation(r:np.ndarray):
    maxPixelValue = np.max(r)
    c = 255 / (np.log(1 + maxPixelValue))        
    result = c * np.log(1 + r)

    return result

# Apply negative on image
def negativeImage(r, L):
    s = L - 1 - r
    return s

# Apply gamma correction on image
def gammaCorrectionImage(r, Y):
    maxPixelValue = np.max(r)
    c = 255 / (maxPixelValue ** Y)
    s = c * r ** Y
    return s

# Function to extract ‘k’ bits from a given position in a number
def extractKBits(num,k,p):
     # Convert number into binary first
     binary = bin(num)
 
     # Remove first two characters
     binary = binary[2:]
 
     end = len(binary) - p
     start = end - k + 1
 
     # Extract k bit sub-string
     kBitSubStr = binary[start : end+1]
 
     # Convert extracted sub-string into decimal again
     return (int(kBitSubStr,2))

# Distances
def getDistance(p, q, mode="euclidean"):
    x, y = p
    u, v = q

    d = 0
    if mode == "euclidean":
        d = np.sqrt((x-u)**2 + (y-v)**2)
    elif mode == "manhattan" or mode == "City block":
        d = np.abs(x-u) + np.abs(y-v)
    elif mode == "Chessboard":
        d = max(np.abs(x-u), np.abs(y-v))

    return d

# Piecewise-Linear Intensity Transformation Functions

def contrastStretching(img, p1, p2):
    r1, s1 = p1
    r2, s2 = p2

    al = s1 / r1
    bt = (s2 - s1) / (r2 - r1)
    gm = (255 - s2) / (255 - r2)
    
    c1 = s1 - bt * r1
    c2 = s2 - gm * r2
    
    d = img
    for i in img.shape[0]:
        for j in img.shape[1]:
            if(img[i][j] < r1):
                d[i][j] = al * img[i][j]
            elif( r1 <= img[i][j] < r2 ):
                d[i][j] = bt * img[i][j] + c1
            else:
                d[i][j] = gm * img[i][j] + c2
    
    return d

def intensityLevelSlicing(mode="two"):
    if mode == "two":
        pass

def bitPlaneSlicing(k):
    pass
