import numpy as np
from math import *

# Constants (Random values) -> to identify the type of popup
INT = 127
FLOAT = 310.47
STR = "STRING"
RADIO = "RADIO"

# Scale function
def scaleImage(image:np.ndarray, mode="scale", a_min=0, a_max=255):
    resultImage = np.zeros(image.shape)        
    
    if mode == "scale":
        image = image - image.min()
        if image.max() == 0 and image.min() == 0:
            resultImage = a_max * (image / 1)   
        else:            
            resultImage = a_max * (image / image.max())   
    elif mode == "clip":
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j] < a_min:
                    resultImage[i,j] = a_min
                elif image[i,j] > a_max:
                    resultImage[i,j] = a_max
                else:
                    resultImage[i,j] = image[i,j]

    resultImage = np.round(np.asarray(resultImage, np.int64))
    return resultImage

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
def getCenter(image:np.ndarray):
    xCenter = image.shape[0] // 2
    yCenter = image.shape[1] // 2
    centerCoordinates = (xCenter,yCenter)
    return centerCoordinates, image[centerCoordinates]

########################
"Statistics"
########################

def getMean(image):
    return np.mean(image)

def getMedian(image):
    return np.median(image)

def getVariance(image):
    return np.var(image)

########################
"Intensity transformations"
########################

# Transform grayscale image to binary
def binaryImage(r:np.ndarray, L=256):
    r = np.round(r / r.max()) * (L-1)
    return r

# Apply negative on image
def negativeImage(r, L=256):
    s = L - 1 - r
    return s

# Log transformation
def logTransformation(r:np.ndarray):
    maxPixelValue = np.max(r)
    c = 255 / (np.log(1 + maxPixelValue))        
    result = c * np.log(1 + r)

    return result

# Apply gamma correction on image
def gammaCorrectionImage(r, Y):
    maxPixelValue = np.max(r)
    c = 255 / (maxPixelValue ** Y)
    s = c * r ** Y
    return s

"Piecewise-Linear Transformation Functions"
# Process that expands the range of intensity
# levels in an image.
def contrastStretching(image, r1, s1, r2, s2):
    # Get shape of image
    row, column = image.shape[0], image.shape[1]

    # Create an zeros array to store the image
    resultImage = np.zeros((row,column))
    for i in range(row):
        for j in range(column):
            p = image[i,j]
            if (0 <= p <= r1):
                resultImage[i,j] = (s1 / r1)*p
            elif (r1 < p <= r2):
                resultImage[i,j] = ((s2 - s1)/(r2 - r1))*(p - r1)+s1
            else:
                resultImage[i,j] = ((255 - s2)/(255 - r2))*(p - r2)+s2
    
    return resultImage

# Highlighting a specific range if intensities 
# in an image often is of interest.
def intensityLevelSlicing(image, A, B, mode="bw"):
    """
    Arguments: 
        - image: image
        - A: min intensity
        - B: max intensity
        - mode: 
            - "bw" -> Black & White
            - "bd" -> Brightness & Darkness
    """

    # Get width and height of image
    row, column = image.shape[0], image.shape[1]

    # Create an zeros array to store the sliced image
    resultImage = np.zeros((row,column))

    # Loop over the input image and if pixel value lies 
    # in desired range set it to 255 otherwise set it to 0.
    for i in range(row):
        for j in range(column):
            if A <= image[i,j] <= B: 
                resultImage[i,j] = 255
            else: 
                if mode == "bd":
                    resultImage[i,j] = image[i,j] 
                else:
                    resultImage[i,j] = 0
    
    return resultImage

# Function to extract ‘k’ bits from a given ‘p’ in a number
def extractKBits(num, k, p):
    # Convert number into binary first
    binary = bin(num)

    # Remove first two characters
    binary = binary[2:]

    end = len(binary) - p
    start = end - k + 1

    # Extract k bit sub-string
    kBitSubStr = binary[start : end+1]
    if kBitSubStr != '':
        # Convert extracted sub-string into decimal again
        return (int(kBitSubStr,2))
    return 0

def bitPlaneSlicing(image, k, p):
    row, column = image.shape[0], image.shape[1]
    resultImage = np.zeros((row,column))
    for i in range(row):
        for j in range(column):
            resultImage[i,j] = extractKBits(int(image[i,j]), k, p)
    return resultImage

########################
"Spatial Transformations"
########################

# Zoom image
def zoom(image, scaleFactor, mode):
    # Get size of original image
    oldWidth, oldHeight = image.shape[0], image.shape[1]

    # Set size of zoomed image
    newWidth = round(oldWidth * scaleFactor)
    newHeight = round(oldHeight * scaleFactor)
    
    # Initialize resized image
    resizedImage = np.zeros([newWidth, newHeight])

    x_ratio = float(oldWidth - 1) / (newWidth - 1) if newWidth > 1 else 0
    y_ratio = float(oldHeight - 1) / (newHeight - 1) if newHeight > 1 else 0

    for i in range(newWidth):
        for j in range(newHeight):
            if mode == "nearest":
                if i/scaleFactor > oldWidth - 1 or j/scaleFactor > oldHeight - 1 :
                    # If I want to know the value of pixel at (3,1) then divide (3/2,1/2) 🡪 int(1.5,0.5) 🡪 (1,0)
                    x = int(i/scaleFactor)
                    y = int(j/scaleFactor)
                else:
                    # If I want to know the value of pixel at (3,1) then divide (3/2,1/2) 🡪 int(1.5,0.5) 🡪 (1,0)
                    x = int(i/scaleFactor)
                    y = int(j/scaleFactor)             
                
                pixel = image[x,y]

            elif mode == "linear":
                x_l, y_l = int(x_ratio * i), int(y_ratio * j)
                x_h, y_h = ceil(x_ratio * i), ceil(y_ratio * j)

                x_weight = (x_ratio * i) - x_l
                y_weight = (y_ratio * j) - y_l
                
                a = image[x_l, y_l]
                b = image[x_l, y_h]
                c = image[x_h, y_l]
                d = image[x_h, y_h]
                    
                pixel = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight
            
            resizedImage[i][j] = pixel
   
    return resizedImage

# Rotate image
def rotate(image, angle, mode="nearest"):
    # Converting degrees to radians
    angle = -angle
    angle = radians(angle)

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
    (centerWidth, centerHeight), _= getCenter(rotatedImage)

    for i in range(oldWidth):
        for j in range(oldHeight):
            x = -(j-centerHeight)*sine + (i-centerWidth)*cosine
            y = (j-centerHeight)*cosine + (i-centerWidth)*sine

            # Add offset
            x += centerWidth
            y += centerHeight
            
            if mode == "nearest":
                # Get nearest index
                x = round(x)
                y = round(y)
                
                # Check if x/y corresponds to a valid pixel in input image
                if (0 <= x < oldWidth and  0 <= y < oldHeight):
                    rotatedImage[i][j] = image[x][y]

            elif mode == "linear":    
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

# Shear image
def shear(image, angle, mode="horizontal"):
    # Converting degrees to radians
    angle = radians(angle)

    # Define the height of the image
    oldWidth = image.shape[0]
    # Define the width of the image
    oldHeight = image.shape[1]
    
    # Initialize rotated image
    shearedImage = np.zeros((oldWidth,oldHeight))
    tangent = tan(-angle)

    # Find the center of the image
    (centerWidth, centerHeight), _ = getCenter(image) # mid col
    
    for i in range(oldWidth):
        for j in range(oldHeight):

            x = (i-centerWidth)
            y = (j-centerHeight)
            
            if mode == "horizontal":
                new_x = round(x-y*tangent)
                new_y = y
            else:
                new_x = x
                new_y = round(y-x*tangent)

            # Add offset
            new_x += centerHeight
            new_y += centerWidth

            if (new_x >= 0 and new_y >= 0 and new_x < oldWidth and new_y < oldHeight):
                shearedImage[j][i] = image[new_y,new_x]
    
    return shearedImage

########################
"Radon Transformations"
########################

# Build the Radon Transform using 'steps' or 'list' of angles projections of 'image'. 
def radon(image, angles, mode='list'):
    ## Accumulate projections in a list.
    projections = []

    if mode == "steps":
        # Angle increment for rotations.
        dTheta = -180.0 / angles
        for i in range(angles):
            rotatedImage = rotate(image, i*dTheta)
            projections.append(rotatedImage.sum(axis=0))
    else:
        for angle in angles:
            rotatedImage = rotate(image, -angle)
            projections.append(rotatedImage.sum(axis=0))
    
    return np.vstack(projections) # Return the projections as a sinogram

########################
"""Filters Functions"""
########################

# Add padding to image
# TODO: Add Replicate padding & Mirror padding
def addPadding(image, paddingSize, mode="zero", value=0):
    if type(paddingSize) == tuple:
        xPaddingSize, yPaddingSize = paddingSize
    else:
        xPaddingSize = paddingSize
        yPaddingSize = paddingSize

    if xPaddingSize < 0:
        xPaddingSize = 0
    if yPaddingSize < 0:
        yPaddingSize = 0
    
    xAddedPadding = 2 * xPaddingSize
    yAddedPadding = 2 * yPaddingSize

    resultImage = np.zeros((image.shape[0] + xAddedPadding, image.shape[1] + yAddedPadding))
    
    if mode == "zero":
        resultImage.fill(value)
        for i in range(xPaddingSize, resultImage.shape[0] - xPaddingSize):
            for j in range(yPaddingSize, resultImage.shape[1] - yPaddingSize):
                resultImage[i][j] = image[i-xPaddingSize][j-yPaddingSize] 
    
    return resultImage

# Convolution function
def convolution(image:np.ndarray, filter:np.ndarray, mode="convolution"):
    if mode == "convolution":
        filter = np.flip(filter)

    filterWidth = filter.shape[0]
    filterHeight = filter.shape[1]
    paddedImage = addPadding(image, (filterWidth//2,filterHeight//2))

    convolvedImage = []
    for i in range(image.shape[0]):
        endPointVertical = i + filterWidth
        rowArray = []
        for j in range(image.shape[1]):
            endPointHorizontal = j + filterHeight
            rowArray.append(np.sum(paddedImage[i:endPointVertical,j:endPointHorizontal] * filter))
        convolvedImage.append(rowArray)
    
    convolvedImage = np.array(convolvedImage)
    return convolvedImage

# Box Kernel
def boxKernel(size:int, shape=None):
    if shape == None:
        shape = (size, size)
        
    filter = np.zeros(shape)
    value = 1/(size*size)
    filter.fill(value)
    
    return filter

# Gaussian Kernel
def gaussianKernel(sigma):
    size = (6*sigma) + 1
    filter = np.zeros((size,size))
    K = 1
    for s in range(size):
        for t in range(size):
            filter[s,t] = K * exp(-(s**2+t**2)/(2*sigma**2))

    return filter

# Order statistics filter (medians & max & min)
def OrderStatisticFilter(image, kernelSize, percent):
    paddingSize = kernelSize // 2
    paddedImage = addPadding(image, paddingSize)

    resultImage = []
    for i in range(image.shape[0]):
        endpointVertical = i + kernelSize
        
        rowArray = []
        for j in range(image.shape[1]):
            endPointHorizontal = j + kernelSize
            rowArray.append(np.percentile(paddedImage[i:endpointVertical,j:endPointHorizontal],percent))

        resultImage.append(rowArray)

    return np.array(resultImage)

###############################################
"""Histogram Functions"""
###############################################

# Build histogram of the image
def getHistogram(image:np.ndarray, bins=256):
    # Calculate the histogram size
    bins = max(image.max(), bins) + 1
    
    # Put pixels in a 1D array by flattening out img array
    flatImage = image.flatten()

    # Array with size of bins, set to zeros
    histogram = np.zeros(bins)
    
    # Loop through pixels and sum up counts of pixels
    for pixel in flatImage:
        histogram[pixel] += 1
    
    # return our final result
    return histogram

# Normalized Histogram
def normalizeHistogram(image:np.ndarray):
    # Calculate max intensity value to equalize to it 
    try:
        L = image.max()
    except:
        L = 256        

    # Get histogram of nonEqualizedImage
    nonEqualizedHistogram = getHistogram(image, bins=L)

    # Normalize
    sumPixels = np.sum(nonEqualizedHistogram)
    nonEqualizedHistogram = nonEqualizedHistogram/sumPixels

    # Normalized cumulative histogram
    cfdHistogram = np.cumsum(nonEqualizedHistogram)

    # Initialized transform map
    transformMap = np.floor((L-1) * cfdHistogram)

    # Flatten image array into 1D list
    flatNonEqualizedImage = list(image.flatten())
    flatEqualizedImage = [transformMap[p] for p in flatNonEqualizedImage]

    # Reshape and write back into equalizedImage
    image = np.reshape(np.asarray(flatEqualizedImage, dtype=np.int64), image.shape)

    return image

# Get mean & variance & std from histogram
def getStatisticsOfHistogram(histogram:np.ndarray, L):
    # Normalize
    sumPixels = np.sum(histogram)
    normalizedHistogram = histogram/sumPixels

    mean = 0
    for i in range(L):
        mean += i * normalizedHistogram[i]
    
    variance = 0
    for i in range(L):
        variance += (i-mean)**2 * normalizedHistogram[i]

    std = sqrt(variance)

    return mean, variance, std
