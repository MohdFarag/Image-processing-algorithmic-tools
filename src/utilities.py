# pylint: disable=C0103,W0105,E0602

"""Algorithms and tools for image processing.
"""

from math import *
import numpy as np
from typing import Tuple, List, Dict, Union, Any

# Constants (Random values) -> to identify the type of popup
INT = 127
FLOAT = 310.47
STR = "STRING"
RADIO = "RADIO"

# Options for radio buttons
DOMAINS = ["spatial", "frequency"]
FILTER_TYPES = ["lowpass", "highpass"]

# Get Data
def getAttribute(variable, attribute):
    if hasattr(variable, attribute):
        # If attribute is found.
        return getattr(variable, attribute)
    else:
        # If attribute is not found.
        return "N/A"

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

# TODO: Test this function
def coordinateIndexing(image:np.ndarray, rows, cols):
    M = image.shape[0]
    if rows * cols != M:
        print("Error: rows * cols != M")
        return None
    
    result = np.zeros((rows, cols))
    for alpha in range(M):
        x = alpha % cols
        y = (alpha-x) / cols
        result[x,y] = image[alpha]
    return result

# TODO: Test this function
def LinearIndexing(image:np.ndarray):
    M = image.shape[0]
    N = image.shape[1]
    
    result = np.zeros(image.shape[0]*image.shape[1])
    for x in range(M):
        for y in range(N):
            alpha = M * y + x
            result[alpha] = image[x,y]
    
def sinusoidal(x, A, f, phi, c=0):
    y = A * np.sin(2*pi*f * x + phi) + c
    return y
###############################################
# "Statistics"
###############################################

def getMean(image):
    return np.mean(image)

def getMedian(image):
    return np.median(image)

def getVariance(image):
    return np.var(image)

###############################################
# "Calculus"
###############################################

def derivative(image): 
    rows = image.shape[0]
    cols = image.shape[1]
    
    # Create a 2D array of zeros
    kernel = np.zeros((rows, cols))
      
    # Apply the Gaussian filter
    for i in range(rows):
        for j in range(cols):
            s = i-(cols//2)
            t = j-(rows//2)
            kernel[i, j] = 0

    return kernel
    
###############################################
# "Set operations"
###############################################

def union2d(image1:np.ndarray, image2:np.ndarray):
    """Union of 2 images

    Args:
        image1 (np.ndarray): input image 1
        image2 (np.ndarray): input image 2

    Returns:
        ndarray: Output image
    """

    row = image1.shape[0]
    col = image1.shape[1]
    resultImage = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            resultImage[i,j] = max(image1[i,j], image2[i,j])

    return resultImage

def intersect2d(image1:np.ndarray, image2:np.ndarray):
    """Intersect between two images

    Args:
        image1 (np.ndarray): input image 1
        image2 (np.ndarray): input image 2

    Returns:
        np.ndarray: Output image
    """

    row = image1.shape[0]
    col = image1.shape[1]
    resultImage = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            resultImage[i,j] = abs(image1[i,j] - image2[i,j])

    return resultImage

def complement2d(image:np.ndarray):
    """Complement of the image

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Output image
    """

    row = image.shape[0]
    col = image.shape[1]
    k = image.max()

    resultImage = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            resultImage[i,j] = k - image[i,j]

    return resultImage

###############################################
# "Intensity transformations"
###############################################

# Transform grayscale image to binary
def binaryImage(image:np.ndarray, L:int=256):
    image = np.round(image / image.max()) * (L-1)
    return image

# Apply negative on image
def negativeImage(image:np.ndarray, L:int=256):
    s = L - 1 - image
    return s

# Log transformation
def logTransformation(image:np.ndarray):
    maxPixelValue = np.max(image)
    c = 255 / (np.log(1 + maxPixelValue))        
    result = c * np.log(1 + image)

    return result

# Apply gamma correction on image
def gammaCorrectionImage(image:np.ndarray, Y:float):
    maxPixelValue = np.max(image)
    c = 255 / (maxPixelValue ** Y)
    s = c * image ** Y
    return s

## "Piecewise-Linear Transformation Functions"

# Process that expands the range of intensity
# levels in an image.
def contrastStretching(image:np.ndarray, r1, s1, r2, s2):
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
# TODO: Test this function
def intensityLevelSlicing(image:np.ndarray, A, B, mode="bw"):
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

def bitPlaneSlicing(image:np.ndarray, k, p):
    row, column = image.shape[0], image.shape[1]
    resultImage = np.zeros((row,column))
    for i in range(row):
        for j in range(column):
            resultImage[i,j] = extractKBits(int(image[i,j]), k, p)
    return resultImage

###############################################
# "Spatial Transformations"
# TODO: Add Bicubic interpolation
###############################################

# Scale image
def scale(image:np.ndarray, scaleFactor:float, mode:str="nearest"):
    # Get size of original image
    oldWidth, oldHeight = image.shape[0], image.shape[1]

    # Set size of scaled image
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
                    x = round(i/scaleFactor)
                    y = round(j/scaleFactor)
                    
                pixel = image[x,y]

            elif mode == "bilinear":
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
def rotate(image:np.ndarray, angle:float, interpolation:str="nearest", option:str='same'):
    '''
    This function rotates the image around its center by amount of degrees
    provided. The rotated image can be of the same size as the original image
    or it can show the full image.

    inputs:
            image: input image (dtype: numpy-ndarray)
            angle: amount of rotation in degrees (e.g., 45,90 etc.)
            interpolation: type of interpolation you need
                'nearest': Nearest neighbour interpolation
                'bilinear' : Bilinear interpolation
            option: string variable for type of rotation. It can take two values
                'same': the rotated image will have same size as the original image
                    It is default value for this variable.
                'full': the rotated image will show the full rotation of original
                    image thus the size may be different than original.
    '''
    
    # Converting degrees to radians
    angle = radians(-angle)

    # Cosine & Sine
    cosine = cos(angle)
    sine = sin(angle)

    # Define the width of the image
    oldWidth = image.shape[0]
    # Define the height of the image
    oldHeight = image.shape[1]
    
    if(option != 'same'):
        # Let us find the height and width of the rotated image
        widthRotated = round(abs(image.shape[0]*cos(angle))) + round(abs(image.shape[1]*sin(angle)))
        heightRotated = round(abs(image.shape[1]*sin(angle))) + round(abs(image.shape[0]*cos(angle)))

        # Initialize rotated image
        rotatedImage = np.zeros((widthRotated, heightRotated))
    else:
        # Initialize rotated image
        rotatedImage = np.zeros((oldWidth,oldHeight))

    # Finding the center point of rotated image.
    (oldCenterWidth, oldCenterHeight), _= getCenter(image)
    (centerWidth, centerHeight), _= getCenter(rotatedImage)

    for i in range(rotatedImage.shape[0]):
        for j in range(rotatedImage.shape[1]):
            if option != "same":
                x = -(j-centerHeight) * sine + (i-centerWidth) * cosine
                y = (j-centerHeight) * cosine + (i-centerWidth) * sine
                # Add offset
                x += oldCenterWidth 
                y += oldCenterHeight
            else:
                x = -(j-oldCenterHeight) * sine + (i-oldCenterWidth) * cosine
                y = (j-oldCenterHeight) * cosine + (i-oldCenterWidth) * sine
                # Add offset
                x += centerWidth
                y += centerHeight

            if interpolation == "nearest":
                # Get nearest index
                x, y = round(x), round(y)
                # Check if x/y corresponds to a valid pixel in input image
                if (0 <= x < oldWidth and  0 <= y < oldHeight):
                    rotatedImage[i][j] = image[x][y]

            elif interpolation == "bilinear":
                # Calculate the coordinate values for 4 surrounding pixels.
                xFloor = int(x)
                xCeil = min(oldWidth-1, ceil(x))
                yFloor = int(y)
                yCeil = min(oldHeight - 1, ceil(y))
                
                if (0 <= x < oldWidth and  0 <= y < oldHeight):
                    if (xCeil == xFloor) and (yCeil == yFloor):
                        q = image[int(x), int(y)]
                    elif (yCeil == yFloor):
                        q1 = image[xFloor, int(y)]
                        q2 = image[xCeil, int(y)]
                        q = q1 * (xCeil - x) + q2 * (x - xFloor)
                    elif (xCeil == xFloor):
                        q1 = image[int(x), yFloor]
                        q2 = image[int(x), yCeil]
                        q = (q1 * (yCeil - y)) + (q2 * (y - yFloor))
                    else:
                        p1 = image[xFloor, yFloor]
                        p2 = image[xFloor, yCeil]
                        p3 = image[xCeil, yFloor]
                        p4 = image[xCeil, yCeil]

                        q1 = p1 * (yCeil - y) + p2 * (y - yFloor)
                        q2 = p3 * (yCeil - y) + p4 * (y - yFloor)
                        q = q1 * (xCeil - x) + q2 * (x - xFloor)
                    
                    rotatedImage[i][j] = q

    return rotatedImage

# Shear image
def shear(image:np.ndarray, angle:float, mode:str="horizontal"):
    """ This function shears the image by amount of degrees

    Args:
        image (ndarray): image to be sheared
        angle (float): amount of shear in degrees
        mode (str, optional): horizontal or vertical . Defaults to "horizontal".

    Returns:
        ndarray: sheared image
    """
    
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

# Translate image
def translate(image:np.ndarray, x:int, y:int):
    """ Translate image by x and y pixels

    Args:
        image (ndarray): Image to translate
        x (int): Number of pixels to translate in x direction
        y (int): Number of pixels to translate in y direction

    Returns:
        ndarray: Translated image
    """
    
    # Define the rows of the image
    rows = image.shape[0]
    # Define the cols of the image
    cols = image.shape[1]
    
    # Initialize translated image
    translatedImage = np.zeros((rows,cols))

    # Find the center of the image
    (centerRows, centerCols), _ = getCenter(image) # mid col
    
    for i in range(rows):
        for j in range(cols):
            new_x = i - x
            new_y = j - y

            if (new_x >= 0 and new_y >= 0 and new_x < rows and new_y < cols):
                translatedImage[j][i] = image[new_y,new_x]
    
    return translatedImage

###############################################
# "Radon Transformations"
###############################################

# Build the Radon Transform using 'steps' or 'list' of angles projections of 'image'.
def radon(image:np.ndarray, angles:Union[int, list, tuple, np.ndarray]):
    """ Build the Radon Transform using 'steps' or 'list' of angles projections of 'image'.

    Args:
        image (np.ndarray): Image to be projected.
        angles (Union[int, list, tuple, np.ndarray]): Angle increment for rotations.

    Returns:
        ndarray: image projections as a sinogram.
    """
    
    ## Accumulate projections in a list.
    projections = []

    if isinstance(angles, int):
        # Angle increment for rotations.
        dTheta = -180.0 / angles
        for i in range(angles):
            rotatedImage = rotate(image, i*dTheta)
            projections.append(rotatedImage.sum(axis=0))
            
    elif isinstance(angles, list) or isinstance(angles, tuple) or isinstance(angles, np.ndarray):
        for angle in angles:
            rotatedImage = rotate(image, -angle)
            projections.append(rotatedImage.sum(axis=0))

    return np.vstack(projections) # Return the projections as a sinogram

# Build a Laminogram of the phantom from sinogram and thetas
def iradon(sinogram, angles=range(180)):
    # Initialize laminogram
    laminogram = np.zeros((sinogram.shape[1],sinogram.shape[1]))
    j = 0
    for i in angles:
        strip = sinogram[i]
        strip = np.tile(strip, (sinogram.shape[1], 1))
        strip = rotate(strip, angles[j])    
        laminogram += strip
        j += 1
    
    return laminogram

###############################################
# "Filters Functions"
###############################################

# Add padding to image
# TODO: Add Replicate padding & Mirror padding
def addPadding(image:np.ndarray, paddingSize, mode="same", value=0):
    if isinstance(paddingSize, tuple):
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
    
    if mode == "same":
        resultImage.fill(value)
        for i in range(xPaddingSize, resultImage.shape[0] - xPaddingSize):
            for j in range(yPaddingSize, resultImage.shape[1] - yPaddingSize):
                resultImage[i][j] = image[i-xPaddingSize][j-yPaddingSize] 
    
    return resultImage

# Convolution function
def convolution(image:np.ndarray, kernel:np.ndarray, mode="convolution"):
    if mode == "convolution":
        kernel = np.flip(kernel)

    kernelWidth = kernel.shape[0]
    kernelHeight = kernel.shape[1]
    paddedImage = addPadding(image, (kernelWidth//2,kernelHeight//2))

    convolvedImage = []
    for i in range(image.shape[0]):
        endPointVertical = i + kernelWidth
        rowArray = []
        for j in range(image.shape[1]):
            endPointHorizontal = j + kernelHeight
            rowArray.append(np.sum(paddedImage[i:endPointVertical,j:endPointHorizontal] * kernel))
        convolvedImage.append(rowArray)
    
    convolvedImage = np.array(convolvedImage)
    return convolvedImage

# Apply specific 'spatial' filter
# You can choose whether the filter applied in spatial or frequency domain
def applySpatialFilter(image:np.ndarray, kernel, domain="spatial"):
    if len(image) != 0:
        if domain == "spatial":
            filteredImage = convolution(image, kernel)
        elif domain == "frequency": 
            rowsSize = image.shape[0] + kernel.shape[0] - 1
            colsSize = image.shape[1] + kernel.shape[1] - 1
            
            # Image fourier
            xImagePadding = (rowsSize - image.shape[0]) // 2
            yImagePadding = (colsSize - image.shape[1]) // 2
            image = addPadding(image, (xImagePadding,yImagePadding))
            
            # Kernel fourier
            xPaddingFilterSize = (rowsSize - kernel.shape[0]) // 2
            yPaddingFilterSize = (colsSize - kernel.shape[1]) // 2
            kernel = addPadding(kernel, (xPaddingFilterSize,yPaddingFilterSize))
            
            size = (min(kernel.shape[0],image.shape[0]), min(kernel.shape[1],image.shape[1]))

            # Apply fourier transform
            grayImageInFreqDomain = fourierTransform(image[:size[0],:size[1]])            
            filterInFreqDomain = fourierTransform(kernel[:size[0],:size[1]])

            # Apply filter in frequency domain
            filteredImageInFreqDomain = filterInFreqDomain * grayImageInFreqDomain

            # Apply inverse fourier transform
            filteredImage = inverseFourierTransform(filteredImageInFreqDomain)
            filteredImage = np.fft.fftshift(filteredImage)
            
            # Get real part
            filteredImage = np.real(filteredImage)
            
            # Remove padding
            filteredImage = filteredImage[xImagePadding:rowsSize-xImagePadding,yImagePadding:colsSize-yImagePadding]
        else:
            print("Domain is not valid")
            return []

        return filteredImage

# Apply specific 'spatial' filter
# You can choose whether the filter applied in spatial or frequency domain
def applyFrequencyFilter(image:np.ndarray, kernel, domain="spatial"):
    if len(image) != 0:
        if domain == "spatial":
            # Apply fourier transform
            imageInFreqDomain = fourierTransform(image)
        elif domain == "frequency": 
            imageInFreqDomain = image
        else:
            print("Domain is not valid")
            return []

        # Apply filter in frequency domain
        filteredImageInFreqDomain = imageInFreqDomain * kernel
        
        # Apply inverse fourier transform
        filteredImage = inverseFourierTransform(filteredImageInFreqDomain)
        
        # Get real part
        filteredImage = np.real(filteredImage)
        return filteredImage

# Box Kernel
def boxFilter(size:int, shape=None):
    if shape == None:
        shape = (size, size)
        
    filter = np.zeros(shape)
    value = 1/(size*size)
    filter.fill(value)
    
    return filter

# Gaussian kernel
def gaussianFilter(sigma:int, size:int=None):
    """Create a Gaussian kernel of size (size x size) and standard deviation sigma.
    """
    
    if size is None:
        # Calculate the size of the kernel
        size = 6*sigma + 1
    
    # Create a 2D array of zeros
    kernel = np.zeros((size, size))
      
    K = 1/(2*np.pi*(sigma**2))
    # Apply the Gaussian filter
    for i in range(size):
        for j in range(size):
            s = i-(size//2)
            t = j-(size//2)
            kernel[i, j] = K*exp(-((s**2)+(t**2))/(2*(sigma**2)))
    
    # Area under the curve should be 1, but the discrete case is only
    # an approximation, correct it
    kernel = kernel / np.sum(kernel)
    
    # Return the kernel
    return kernel

# Gaussian low pass filter
def gaussianLowPassFilter(shape, d0=9):
    """ Gaussian low pass filter

    Args:
        shape (tuple): Shape of the filter
        d0 (int, optional): Diameter (cutoff frequency) of the filter. Defaults to 9.

    Returns:
        ndarray: Filter
    """
    kernel = np.zeros(shape)
    rows, cols = shape
    
    centerRow = rows // 2
    centerCol = cols // 2
    
    for i in range(rows):
        for j in range(cols):
            kernel[i][j] = exp(-((i - centerRow)**2 + (j - centerCol)**2) / (2 * d0**2))
                
    return kernel

# Ideal low pass filter
def idealLowPassFilter(shape, d0=9):
    """ Ideal low pass filter

    Args:
        shape (tuple): Shape of the filter
        d0 (int, optional): Diameter of the filter. Defaults to 9.

    Returns:
        ndarray: Low Pass Filter
    """
    kernel = np.zeros(shape)
    rows, cols = shape
    
    centerRow = rows // 2
    centerCol = cols // 2
    
    for i in range(rows):
        for j in range(cols):
            if sqrt((i - centerRow)**2 + (j - centerCol)**2) <= d0:
                kernel[i][j] = 1
                
    return kernel

# Butterworth low pass filter
def butterworthLowPassFilter(shape, d0=9, n=2):
    """ Butterworth low pass filter

    Args:
        shape (tuple): Shape of the filter
        n (int): Order of the filter
        d0 (int, optional): Diameter (cutoff frequency) of the filter. Defaults to 9.

    Returns:
        ndarray: Filter
    """
    kernel = np.zeros(shape)
    rows, cols = shape
    
    centerRow = rows // 2
    centerCol = cols // 2
    
    for i in range(rows):
        for j in range(cols):
            kernel[i][j] = 1 / (1 + (sqrt((i - centerRow)**2 + (j - centerCol)**2) / d0)**(2*n))
                
    return kernel

# Laplacian filter
def laplacianFilter(domain='frequency', shape=(3,3), enhance=False):
    if domain == 'spatial':
        kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    elif domain == 'frequency':
        kernel = np.zeros(shape)
        rows, cols = shape
        
        centerRow = rows // 2
        centerCol = cols // 2
        
        for i in range(rows):
            for j in range(cols):
                if not enhance:
                    kernel[i][j] = -4 * pi**2 * ((i - centerRow)**2 + (j - centerCol)**2)
                else:
                    kernel[i][j] = 1 + 4 * pi**2 * ((i - centerRow)**2 + (j - centerCol)**2)

        return kernel

# Ideal high pass filter
def idealHighPassFilter(shape, d0=9):
    """ Ideal high pass filter

    Args:
        shape (tuple): Shape of the filter
        d0 (int, optional): Diameter of the filter. Defaults to 9.

    Returns:
        ndarray: High pass Filter
    """
    kernel = np.ones(shape)
    rows, cols = shape
    
    centerRow = rows // 2
    centerCol = cols // 2
    
    for i in range(rows):
        for j in range(cols):
            if sqrt((i - centerRow)**2 + (j - centerCol)**2) <= d0:
                kernel[i][j] = 0
                
    return kernel

# Butterworth high pass filter
def butterworthHighPassFilter(shape, d0=9, n=2):
    """ Butterworth high pass filter

    Args:
        shape (tuple): Shape of the filter
        n (int): Order of the filter
        d0 (int, optional): Diameter (cutoff frequency) of the filter. Defaults to 9.

    Returns:
        ndarray: Filter
    """
    kernel = np.zeros(shape)
    rows, cols = shape
    
    centerRow = rows // 2
    centerCol = cols // 2
    
    for i in range(rows):
        for j in range(cols):
            kernel[i][j] = 1 / (1 + (d0 / sqrt((i - centerRow)**2 + (j - centerCol)**2))**(2*n))
                
    return kernel

# Gaussian high pass filter
def gaussianHighPassFilter(shape, d0=9):
    """ Gaussian high pass filter

    Args:
        shape (tuple): Shape of the filter
        d0 (int, optional): Diameter (cutoff frequency) of the filter. Defaults to 9.

    Returns:
        ndarray: Filter
    """
    kernel = np.zeros(shape)
    rows, cols = shape
    
    centerRow = rows // 2
    centerCol = cols // 2
    
    for i in range(rows):
        for j in range(cols):
            kernel[i][j] = 1 - exp(-((i - centerRow)**2 + (j - centerCol)**2) / (2 * d0**2))
                
    return kernel

# Homomorphic filter
def homomorphicFilter(shape, d0=9, c=1, gammaLow=0.5, gammaHigh=1.5):
    """ Homomorphic filter

    Args:
        shape (tuple): Shape of the filter
        d0 (int, optional): Diameter (cutoff frequency) of the filter. Defaults to 9.
        c (int, optional): Constant. Defaults to 1.
        gammaLow (float, optional): Low gamma. Defaults to 0.5.
        gammaHigh (float, optional): High gamma. Defaults to 1.5.

    Returns:
        ndarray: Filtered image
    """

    kernel = np.zeros(shape)
    rows, cols = shape
    
    centerRow = rows // 2
    centerCol = cols // 2
    
    for i in range(rows):
        for j in range(cols):
            D = sqrt((i - centerRow)**2 + (j - centerCol)**2)
            kernel[i][j] = (gammaHigh - gammaLow) * (1 - exp(-c * D**2  / d0**2)) + gammaLow
                
    return kernel
           
# Order statistics filter (medians & max & min)
def orderStatisticFilter(image:np.ndarray, kernelSize:int, percent):
    paddingSize = kernelSize // 2
    paddedImage = addPadding(image, paddingSize)

    resultImage = []
    for i in range(image.shape[0]):
        endpointVertical = i + kernelSize
        
        rowArray = []
        for j in range(image.shape[1]):
            endPointHorizontal = j + kernelSize
            rowArray.append(np.percentile(paddedImage[i:endpointVertical,j:endPointHorizontal], percent))

        resultImage.append(rowArray)

    return np.array(resultImage)

# Band reject filter
def bandRejectFilter(image:np.ndarray, magnitudeSpectrum, points, d0=9):
    if len(image) != 0:
        n = magnitudeSpectrum.shape[0]
        m = magnitudeSpectrum.shape[1]
        for u in range(n):
            for v in range(m):
                for d in range(len(points)):
                    v0 = points[d][0]
                    u0 = points[d][1]
                    d1 = (u - u0)**2 + (v - v0)**2
                    d2 = (u + u0)**2 + (v + v0)**2
                    if d1 <= d0 or d2 <= d0:
                        magnitudeSpectrum[u][v] = 0
    
        resultImage = np.abs(inverseFourierTransform(magnitudeSpectrum))
        return resultImage

# Perform un-sharp masking
def unsharpMask(image, size, k, domain='spatial', filterType='lowpass'):
    if domain == 'spatial':
        kernel = boxFilter(size)
        # Apply box kernel
        blurredImage = applySpatialFilter(image, kernel)
        # Subtract blurred image from original image
        mask = image - blurredImage
        # Multiply the result by k (highboost factor) then sum to original image
        resultImage = image + k * mask   
        
    elif domain == 'frequency':
        if filterType == 'highpass':
            k1 = 1
            k2 = k
            kernel = idealHighPassFilter(image.shape, size)
        elif filterType == 'lowpass':
            k1 = 1 - k
            k2 = -k
            kernel = idealLowPassFilter(image.shape, size)
            
        imageInFreqDomain = fourierTransform(image)
        
        # Get filtered image in frequency domain
        filteredImageInFreqDomain = imageInFreqDomain * (k1 + k2 * kernel)
        resultImage = np.real(inverseFourierTransform(filteredImageInFreqDomain))
        
    return resultImage

###############################################
# "Histogram Functions"
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
def normalizeHistogram(image: np.ndarray):
    # Calculate max intensity value to equalize to it 
    L = image.max()      

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
    equalizedImage = np.reshape(flatEqualizedImage, image.shape)

    return equalizedImage

# Get mean & variance & std from histogram
def getStatisticsOfHistogram(histogram:np.ndarray, L=256):
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

###############################################
# "Morphological Transformations"
###############################################

# Erode the image
def erosion(image:np.ndarray, SE=None):
    # Define the structuring element
    if SE is None:
        k = 3
        SE = np.ones((k,k))
    else:
        k = len(SE)
    constant = (k-1) // 2

    # Acquire size of the image
    m, n= image.shape 
    
    # Define new image
    AdB = np.zeros((m,n))
    
    # Erosion without using inbuilt cv2 function for morphology
    for i in range(constant, m-constant):
        for j in range(constant,n-constant):
            temp = image[i-constant:i+constant+1, j-constant:j+constant+1]
            product = morphologyMultBitWise(temp,SE,1)
            AdB[i,j] = np.min(product)
    return AdB

# Dilate the image
def dilation(image:np.ndarray, SE=None):
    # Define the structuring element
    if SE is None:
        k = 3
        SE = np.ones((k,k))
    else:
        k = len(SE)
    constant = (k-1)//2
    
    # Acquire size of the image
    m, n = image.shape
    
    # Define new image to store the pixels of dilated image
    AdB = np.zeros((m,n))

    # Dilation operation
    for i in range(constant, m-constant):
        for j in range(constant,n-constant):
            temp = image[i-constant:i+constant+1, j-constant:j+constant+1]
            product = morphologyMultBitWise(temp,SE,0)
            AdB[i,j] = np.max(product)

    return AdB

# Opening the image
def opening(image:np.ndarray, SE=None):
    # Erode the image
    AeB = erosion(image, SE)
    # Dilate the eroded image
    AoB = dilation(AeB, SE)

    return AoB

# Closing the image
def closing(image:np.ndarray, SE=None):
    # Dilate the image
    AdB = dilation(image, SE)
    # Erode the dilated image
    AcB = erosion(AdB, SE)

    return AcB

# Hit or miss transform
def hitOrMissTransform(image:np.ndarray, B1:np.ndarray, B2:np.ndarray):
    AoB1 = erosion(image, B1)
    AcoB2 = erosion(np.bitwise_not(image), B2)
    AhmtB = np.bitwise_and(AoB1,AcoB2)

    return AhmtB

# Boundary extraction
def boundaryExtraction(image:np.ndarray, SE=None):
    return image - erosion(image,SE)

# Multiplication Bitwise
def morphologyMultBitWise(temp:np.ndarray,SE:np.ndarray, defaultValue=1):
    if temp.shape == SE.shape:
        size = temp.shape
        result = np.zeros(size)
        result.fill(defaultValue)

        for i in range (size[0]):
            for j in range(size[1]):
                if SE[i][j] != 0 and SE[i][j] is not None:
                    result[i][j] = temp[i][j] * SE[i][j]

        return result
    else:
        print("**Error** first matrix shape not equal second matrix shape")

###############################################
# "Fourier Transformations"
###############################################

# Fourier transform
def fourierTransform(image:np.ndarray, mode="fourier", log=False):
    f = np.fft.fft2(image)
    fShift = np.fft.fftshift(f)
    spectrum = None

    if mode == "magnitude":
        if not log:
            spectrum = np.abs(fShift)
        else:
            spectrum = logTransformation(np.abs(fShift))
    elif mode == "phase":
        spectrum  = np.angle(fShift)
    else:
        spectrum = fShift

    return spectrum

# inverse Fourier transform
def inverseFourierTransform(combinedImage:Union[np.ndarray,tuple], mode:str="normal"):
    """ Inverse Fourier transform

    Args:
        combinedImage (np.ndarray or tuple): Image to be transformed
        mode (str): Mode of the image. "normal" for normal image, "separate" for magnitude and phase image

    Returns:
        np.ndarray: Inverse Fourier transformed image
    """
    
    if mode=="separate":
        combinedImage = np.multiply(combinedImage[0], np.exp(1j * combinedImage[1]))
    
    shiftedImage = np.fft.ifftshift(combinedImage)
    resultImage = np.fft.ifft2(shiftedImage)
    return resultImage

###############################################
# "Noise"
###############################################

# Add noise to the image
def addNoise(image, noise, scale=False):
    if scale:
        noise = scaleImage(noise, "scale", 0, 255)
    else:
        noise = np.round(np.asarray(noise, np.int64))

    image = image + noise

    return image

# Add salt & pepper noise to the image
def addSaltAndPepperNoise(image, mode="salt and pepper"):
    rows, cols = image.shape

    # Randomly pick some pixels in the image for coloring them white
    number_of_pixels = int((np.random.randint(2,7)/100) * (rows*cols))

    if mode == "salt and pepper":
        salt = True
        pepper = True
    elif mode == "salt":
        salt = True
        pepper = False
    elif mode == "pepper":
        salt = False
        pepper = True

    if pepper is True:
        for _ in range(number_of_pixels):        
            image[np.random.randint(0, rows - 1)][np.random.randint(0, cols - 1)] = 255

    if salt is True:
        for _ in range(number_of_pixels):        
            image[np.random.randint(0, rows - 1)][np.random.randint(0, cols - 1)] = 0

    return image