# math & matrix computations library
import math
import numpy as np
import random

# Matplotlib
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Phantom
from phantominator import shepp_logan
from skimage.transform import rotate ## Image rotation routine
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

from PIL import Image
from .utilities import *

# Pydicom
import pydicom as dicom

class ImageViewer(FigureCanvasQTAgg):
    def __init__(self, parent=None, axisExisting=False, axisColor="#329da8", type="image", title=""):
        self.fig = Figure(figsize = (6, 6), dpi = 80)
        self.axes = self.fig.add_subplot(111)

        # Variables
        self.loaded = False
        self.originalImage = np.array([])
        self.grayImage = np.array([])
        self.axisExisting = axisExisting
        self.axisColor = axisColor
        self.title = title
        self.axes.grid(False)

        if type == "image":
            self.xlabel = "Width"
            self.ylabel = "Height"
        
        elif type == "hist":
            self.xlabel = "Intensity"
            self.ylabel = "Count"
            divider = make_axes_locatable(self.axes)
            cax = divider.append_axes('bottom', size='6%', pad=0.55, add_to_figure=True)
            
            cmap = mpl.cm.gray
            norm = mpl.colors.Normalize(vmin=0, vmax=255)
            
            self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
        
        self.axes.set_title(self.title)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.setTheme()

        super(ImageViewer, self).__init__(self.fig)
   
    # Set grid
    def setGrid(self, status):
        self.axes.grid(status)

    # Set Theme
    def setTheme(self):
        fontStyle = {'fontsize': 17,
                    'fontweight' : 900,
                    'verticalalignment': 'top'}
        self.axes.set_title(self.title,fontStyle)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)

        self.fig.set_edgecolor("black")
        self.axes.spines['bottom'].set_color(self.axisColor)
        self.axes.spines['top'].set_color(self.axisColor)
        self.axes.spines['right'].set_color(self.axisColor)
        self.axes.spines['left'].set_color(self.axisColor)

        if not self.axisExisting:
            self.axes.set_xticks([])
            self.axes.set_yticks([])

    ###############################################
    """Image Functions"""
    ###############################################

    # Set image
    def setImage(self, image_path, fileExtension):
        # Reading the image
        if fileExtension == "dcm":
            imgData = dicom.dcmread(image_path, force=True)
            self.originalImage = imgData.pixel_array
        else:
            self.originalImage = mpimg.imread(image_path)
            imgData = Image.open(image_path)

        # If image is RGB transform it to gray.

        if self.originalImage.ndim > 2:
            self.grayImage = self.originalImage[:,:,0]
        else:
            self.grayImage = self.originalImage

        self.grayImage = self.scaleImage(self.grayImage)
        self.loaded = True

        self.drawImage(self.grayImage)

        # Depends on extension of the file
        return imgData

    # Get image
    def getGrayImage(self):
        return self.grayImage
    
    # Get original image pixels
    def getOriginalImage(self):
        return self.originalImage
    
    # Draw image with matplotlib
    def drawImage(self, image, title="Blank", cmap="gray"):
        self.axes.set_title(title, fontsize = 16)
        self.axes.imshow(image, cmap=cmap)
        self.draw()

    # Scale function
    def scaleImage(self, image, mode="scale", a_min=0, a_max=255):
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

    # Save Image
    def saveImage(self,path):
        self.fig.savefig(path, bbox_inches='tight')

    # Clear figure
    def clearImage(self):
        self.axes.clear()
        self.setTheme()
        self.draw()

    # Reset figure and variables
    def reset(self):
        self.clearImage()

        self.loaded = False
        self.grayImage = np.array([])
    
    ###############################################
    """plt Functions"""
    ###############################################

    def setROI(self, corners):
        corners = np.int64(np.round(corners))
        p1 = corners[0][0], corners[1][0]
        p2 = corners[0][2], corners[1][2]
        ROI = self.grayImage[p1[1]:p2[1], p1[0]:p2[0]]
        self.drawImage(ROI)

        L = ROI.max()
        # Get histogram of image
        histogram = np.bincount(ROI.flatten(), minlength=L)
        mean, variance, std = self.getStatOfHist(histogram, L)

        self.grayImage = ROI
        return mean, variance, std

    ###############################################
    """Image Functions"""
    ###############################################

    # Draw logged image
    def logImage(self):
        self.grayImage = logTransformation(self.grayImage)
        self.drawImage(self.grayImage)

    # Apply negative on image
    def negativeImage(self):
        self.grayImage = negativeImage(self.grayImage, 256)
        self.drawImage(self.grayImage)

    # Apply gamma correction on image
    def gammaCorrectionImage(self, Y):
        self.grayImage = gammaCorrectionImage(self.grayImage, Y)
        self.drawImage(self.grayImage)

    ###############################################
    """Piecewise-Linear Intensity Transformation Functions"""
    ###############################################

    def contrastStretching(self, r, s):
        r1, r2 = r
        s1, s2 = s
    
    def intensityLevelSlicing(self, mode="two"):
        pass

    def bitPlaneSlicing(self, k):
        pass

    ###############################################
    """Transformations Functions"""
    ###############################################

    # Zoom image
    def zoomImage(self, scaleFactor, mode):
        if self.loaded and self.grayImage.ndim == 2:
            scaleFactor = float(scaleFactor)
            # Get size of original image
            oldWidth = self.grayImage.shape[0]
            oldHeight = self.grayImage.shape[1]
            # Set size of zoomed image
            newWidth = round(oldWidth * scaleFactor)
            newHeight = round(oldHeight * scaleFactor)
            # Initialize resized image
            resizedImage = np.zeros([newWidth, newHeight])

            if mode == "nearest":
                # Set the values
                for i in range(newWidth):
                    for j in range(newHeight):
                        if i/scaleFactor > oldWidth - 1 or j/scaleFactor > oldHeight - 1 :
                            # If I want to know the value of pixel at (3,1) then divide (3/2,1/2) 🡪 floor(1.5,0.5) 🡪 (1,0)
                            x = math.floor(i/scaleFactor)
                            y = math.floor(j/scaleFactor)
                        else:
                            # If I want to know the value of pixel at (3,1) then divide (3/2,1/2) 🡪 floor(1.5,0.5) 🡪 (1,0)
                            x = round(i/scaleFactor)
                            y = round(j/scaleFactor)

                        resizedImage[i,j] = self.grayImage[x,y]
            
            elif mode == "linear":
                y_ratio = float(oldWidth - 1) / (newWidth - 1) if newWidth > 1 else 0
                x_ratio = float(oldHeight - 1) / (newHeight - 1) if newHeight > 1 else 0

                for i in range(newWidth):
                    for j in range(newHeight):
                        x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
                        x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

                        x_weight = (x_ratio * j) - x_l
                        y_weight = (y_ratio * i) - y_l
                        
                        a = self.grayImage[y_l, x_l]
                        b = self.grayImage[y_l, x_h]
                        c = self.grayImage[y_h, x_l]
                        d = self.grayImage[y_h, x_h]
                            
                        pixel = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight
                        resizedImage[i][j] = pixel
        else:
            return "N/A","N/A"

        zoomWidth = int(resizedImage.shape[0]/scaleFactor)
        zoomHeight = int(resizedImage.shape[1]/scaleFactor)
        
        self.grayImage = self.scaleImage(resizedImage[:zoomWidth,:zoomHeight])
        self.drawImage(self.grayImage)

        return resizedImage.shape[0], resizedImage.shape[1]

    # Rotate image
    def rotateImage(self, image, angle, mode, output="size"):
        if len(image) != 0:
            # Converting degrees to radians
            angle = -angle
            angle = math.radians(angle)

            # Cosine & Sine
            cosine = math.cos(angle)
            sine = math.sin(angle)

            # Define the height of the image
            oldWidth = image.shape[0]
            # Define the width of the image
            oldHeight = image.shape[1]

            # Define the width and height of the new image that is to be formed
            newWidth = round(abs(image.shape[0]*cosine)+abs(image.shape[1]*sine))+1
            newHeight = round(abs(image.shape[1]*cosine)+abs(image.shape[0]*sine))+1
         
            # Initialize rotated image 
            # rotatedImage = np.zeros((oldWidth,oldHeight)) 
            rotatedImage = np.zeros((newWidth,newHeight)) 

            # Find the center of the rotated image
            (centerWidth, centerHeight), _= getCenter(rotatedImage)

            for i in range(newWidth):
                for j in range(newHeight):
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
                        x_floor = math.floor(x)
                        x_ceil = min(oldWidth-1, math.ceil(x))
                        y_floor = math.floor(y)
                        y_ceil = min(oldHeight - 1, math.ceil(y))
                        
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

            if output == "size":
                self.grayImage = self.scaleImage(rotatedImage)
                self.drawImage(self.grayImage)

                return rotatedImage.shape[0],rotatedImage.shape[1]
            
            elif output == "image":
                return self.grayImage
        else:
            return "N/A","N/A"

    # Shear image
    def shearImage(self, angle, mode="horizontal"):
        if self.loaded:
            # Converting degrees to radians
            angle = math.radians(angle)

            # Define the height of the image
            oldWidth = self.grayImage.shape[0]
            # Define the width of the image
            oldHeight = self.grayImage.shape[1]
          
            # Initialize rotated image
            shearedImage = np.zeros((oldWidth,oldHeight))
            tangent = math.tan(-angle)

            # Find the center of the image
            (centerWidth, centerHeight), _ = getCenter(self.grayImage) # mid col
            
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
                        shearedImage[j][i] = self.grayImage[new_y,new_x]

            self.grayImage = self.scaleImage(shearedImage)
            self.drawImage(self.grayImage)

    ###############################################
    """Shapes construction Functions"""
    ###############################################

    # Construct T shape
    def constructT(self):
        self.grayImage = np.zeros((128,128), dtype=np.int64)
        
        for i in range(29,50):
            for j in range(29,100):
                self.grayImage[i,j] = 255
        
        for i in range(49,100):
            for j in range(54,74):
                self.grayImage[i,j] = 255

        self.loaded = True
        self.drawImage(self.grayImage)

    # Construct Triangle shape
    def constructTriangle(self, background="white"):
        self.grayImage = np.zeros((128,128), dtype=np.int64)
        
        k = 100 - 29
        for i in range(29,100):
            for j in range(k,127-k):
                self.grayImage[i,j] = 255
            k -= 1

        self.loaded = True
        self.drawImage(self.grayImage)

    # Construct Circle in gray box shape
    def constructCircle(self, I1=250, I2=150, I3=50, d=184, r=64):
        # Parameters
        n = 256
        m = 256
        self.grayImage = np.zeros((n,m), dtype=np.int64)
        (centerX, centerY), _ = getCenter(self.grayImage)
        differenceSize = (n - d) // 2

        for i in range(n):
            for j in range(m):
                if (differenceSize <= i <= n-differenceSize) and (differenceSize <= j <= m-differenceSize):
                    if (i - centerX)**2 + (j - centerY)**2 < r**2:
                        self.grayImage[i,j] = I1
                    else:
                        self.grayImage[i,j] = I2
                else:
                    self.grayImage[i,j] = I3

        self.loaded = True
        self.drawImage(self.grayImage)

    # Construct Square shape
    def constructSquare(self, background=255):
        # Parameters
        n,m = 128,128
        length = 64

        self.grayImage = np.zeros((n,m), dtype=np.int64)
        differenceSize = (n - length) // 2

        for i in range(n):
            for j in range(m):
                if (differenceSize <= i <= n-differenceSize) and (differenceSize <= j <= m-differenceSize):
                    self.grayImage[i,j] = background
                else:
                    self.grayImage[i,j] = 0

        self.loaded = True
        self.drawImage(self.grayImage)

    # Construct Background shape
    def constructBackground(self, background=0):
        self.grayImage = np.zeros((128,128), dtype=np.int64)
        
        self.grayImage.fill(background)

        self.loaded = True
        self.drawImage(self.grayImage)

    ################
    """Phantom"""
    ################

    # Construct phantom
    def constructPhantom(self):
        phantom=np.zeros((256,256))
        for i in range(118,138):
            for j in range(118,138):
                phantom[i][j] =255 

        # phantom = shepp_logan(256)
        # phantom = np.flip(phantom)
                
        self.grayImage = self.scaleImage(phantom)
        
        self.loaded = True
        self.drawImage(self.grayImage)

    # Build the Radon Transform using 'steps' projections of 'image'. 
    def radon(self, image, steps):                
        ## Accumulate projections in a list.
        projections = []
        # Angle increment for rotations.
        dTheta = -180.0 / steps

        for i in range(steps):
            projections.append(rotate(image, i*dTheta).sum(axis=0))
        
        return np.vstack(projections) # Return the projections as a sinogram
        
    # Display a Sinogram of this phantom
    def drawSinogram(self, image):
        sinogram = radon(image)
        # sinogram = self.radon(image, 180)
        self.drawImage(sinogram, "Sinogram")

    # Display a Laminogram of this phantom
    def drawLaminogram(self, image):
        laminogram = np.zeros((image.shape[1], image.shape[1]))
        dTheta = 180.0 / image.shape[0]

        # thetaGroup = range(image.shape[0])
        thetaGroup = [0, 20, 40, 60]
        for i in thetaGroup:
            temp = np.tile(image[i],(image.shape[1],1))
            temp = rotate(temp, i)
            laminogram += temp

        self.drawImage(laminogram, "Laminogram")

    ###############################################
    """Histogram Functions"""
    ###############################################

    # Build histogram of the image
    def getHistogram(self, image:np.ndarray, bins):
        # Put pixels in a 1D array by flattening out img array
        flatImage = image.flatten()

        # Array with size of bins, set to zeros
        histogram = np.zeros(bins)
        
        # Loop through pixels and sum up counts of pixels
        for pixel in flatImage:
            histogram[pixel] += 1
        
        # return our final result
        return histogram

    # Get histogram image
    def drawHistogram(self, image:np.ndarray):
        self.clearImage()

        if len(image) != 0:
            L = image.max()
          
            # Get histogram of image
            histogram = np.bincount(image.flatten(), minlength=L)

            # Calculate statistics of histogram
            mean, variance, std = self.getStatOfHist(histogram, L)
            
            # Normalize
            sumPixels = np.sum(histogram)
            normalizedHistogram = histogram/sumPixels

            cdf = normalizedHistogram.cumsum()
            cdf_normalized = cdf * float(normalizedHistogram.max()) / cdf.max()

            self.axes.bar(range(len(normalizedHistogram)), normalizedHistogram, color='red')
            self.axes.plot(cdf_normalized, color = 'black')
            self.axes.axvline(mean, color='g', linestyle='dashed', linewidth=1)
            self.axes.legend(('cumulative histogram','mean','histogram'), loc = 'upper left')

            self.axes.set_title(f'\u03BC = {mean:.4}    \u03C3 = {std:.4}')
            self.draw()
        else:
            return

    # Normalized Histogram
    def normalizeHistogram(self):
        if len(self.grayImage) != 0:
            self.clearImage() # Clear prev.
            
            # Calculate max intensity value to equalize to it 
            try:
                L = self.grayImage.max()
            except:
                L = 256        

            # Get histogram of nonEqualizedImage
            nonEqualizedHistogram = np.bincount(self.grayImage.flatten(), minlength=L)

            # Normalize
            sumPixels = np.sum(nonEqualizedHistogram)
            nonEqualizedHistogram = nonEqualizedHistogram/sumPixels

            # Normalized cumulative histogram
            cfdHistogram = np.cumsum(nonEqualizedHistogram)

            # Initialized transform map
            transformMap = np.floor((L-1) * cfdHistogram)

            # Flatten image array into 1D list
            flatNonEqualizedImage = list(self.grayImage.flatten())
            flatEqualizedImage = [transformMap[p] for p in flatNonEqualizedImage]

            # Reshape and write back into equalizedImage
            self.grayImage = np.reshape(np.asarray(flatEqualizedImage, dtype=np.int64), self.grayImage.shape)

            # Draw equalized image
            self.drawImage(self.grayImage)
        else:
            return

    def getStatOfHist(self, histogram:np.ndarray, L):
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
    """Filters Functions"""
    ###############################################

    # Add padding to image
    def addPadding(self, image, paddingSize, mode="same", value=0):
        if type(paddingSize) == tuple:
            xPaddingSize, yPaddingSize = paddingSize
        else:
            xPaddingSize = paddingSize
            yPaddingSize = paddingSize
        
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
    def convolution(self, filter:np.ndarray, mode="convolution"):
        if mode == "convolution":
            filter = np.flip(filter)

        filterWidth = filter.shape[0]
        filterHeight = filter.shape[1]
       
        filterSize = max(filterWidth, filterHeight)
        paddingSize = filterSize // 2

        paddedImage = np.pad(self.grayImage, paddingSize) 
        # paddedImage = self.addPadding(self.grayImage, paddingSize)

        convolvedImage = []
        for i in range(self.grayImage.shape[0]):
            endPointVertical = i + filterSize
            rowArray = []
            for j in range(self.grayImage.shape[1]):
                endPointHorizontal = j + filterSize
                rowArray.append(np.sum(paddedImage[i:endPointVertical,j:endPointHorizontal] * filter))
            convolvedImage.append(rowArray)
        
        convolvedImage = np.array(convolvedImage)

        return convolvedImage

    # Box Kernel
    def boxKernel(self, size:int, shape=None):
        if shape == None:
            shape = (size, size)
            
        value = 1/(size*size)

        filter = np.ones(shape)
        filter.fill(value)
        
        return filter
    
    # Subtract blurred image from original
    def subtractBlurredFromOriginal(self, blurredImage):
        resultImage = np.subtract(self.grayImage, blurredImage)
        return resultImage
    
    # Multiply by a factor K Then added to the original image
    def multiplyByFactor(self, image, k):
        resultImage = np.multiply(image,k)
        resultImage = np.add(resultImage, self.grayImage)

        return resultImage
    
    # Median mask
    def medianMask(self, size):
        if len(self.grayImage) != 0:
            filterSize = size
            paddingSize = filterSize // 2
            paddedImage = self.addPadding(self.grayImage, paddingSize)

            resultImage = []
            for i in range(self.grayImage.shape[0]):
                endpointVertical = i + filterSize
                
                rowArray = []
                for j in range(self.grayImage.shape[1]):
                    endPointHorizontal = j + filterSize
                    rowArray.append(np.median(paddedImage[i:endpointVertical,j:endPointHorizontal]))

                resultImage.append(rowArray)
            
            scaledImage = self.scaleImage(np.array(resultImage))
            self.grayImage = scaledImage
            # Draw image
            self.drawImage(self.grayImage)

    # Perform un-sharp masking
    def unsharpMask(self, size, k):
        if len(self.grayImage) != 0:
            self.clearImage()
            boxFilter = self.boxKernel(size)
            blurredImage = self.applySpatialFilter(self.grayImage, boxFilter)

            subtractedImage = self.subtractBlurredFromOriginal(blurredImage)
            resultImage = self.multiplyByFactor(subtractedImage, k)

            scaledImage = self.scaleImage(resultImage)
            self.grayImage = scaledImage
            
            # Draw image
            self.drawImage(self.grayImage)

    # Apply specific filter
    def applySpatialFilter(self, image, kernel, domain="spatial"):
        if len(self.grayImage) != 0:
            rows = self.grayImage.shape[0] + kernel.shape[0] - 1
            cols = self.grayImage.shape[1] + kernel.shape[1] - 1
            size = (rows,cols)

            xImagePadding = kernel.shape[0] // 2
            yImagePadding = kernel.shape[1] // 2
            xPaddingFilterSize = self.grayImage.shape[0] // 2
            yPaddingFilterSize = self.grayImage.shape[1] // 2

            blurredImage = np.array([])
            if domain == "spatial":
                blurredImage = self.convolution(kernel)
            
            elif domain == "frequency": 
                # Image fourier
                image = self.addPadding(image, (xImagePadding,yImagePadding))               
                grayImageInFreqDomain = self.fourierTransform(image, draw=False, s=size)
                
                # Kernel fourier
                kernel = self.addPadding(kernel, (xPaddingFilterSize,yPaddingFilterSize))
                boxFilterInFreqDomain = self.fourierTransform(kernel, draw=False,s=size)

                filteredImageInFreqDomain = boxFilterInFreqDomain * grayImageInFreqDomain

                blurredImage = self.inverseFourierTransform(filteredImageInFreqDomain)
                blurredImage = np.fft.fftshift(blurredImage)

                blurredImage = np.abs(blurredImage)
                blurredImage = blurredImage[xImagePadding:rows-xImagePadding,yImagePadding:cols-yImagePadding]
                
            blurredImage = self.scaleImage(blurredImage)

            return blurredImage
    
    # Perform box filtering
    def boxFiltering(self, size):
        if len(self.grayImage) != 0:
            self.clearImage()
            boxFilter = self.boxKernel(size)
            self.grayImage = self.applySpatialFilter(self.grayImage, boxFilter)
                       
            # Draw image
            self.drawImage(self.grayImage)

    # Perform box filtering in frequency domain
    def boxFilteringUsingFourier(self, filterSize):
        if len(self.grayImage) != 0:
            self.clearImage()
            boxFilter = self.boxKernel(filterSize)

            self.grayImage = self.applySpatialFilter(self.grayImage, boxFilter, domain="frequency")
            
            # Draw image
            self.drawImage(self.grayImage)

    # Remove periodic noise
    def notchRejectFilters(self, magnitudeSpectrum, points, d0=9):
        if len(self.grayImage) != 0:
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
        
            resultSpectrum = np.abs(self.inverseFourierTransform(magnitudeSpectrum))
            self.grayImage = self.scaleImage(resultSpectrum)

            # Draw image
            self.drawImage(self.grayImage)
            
    ###############################################
    """Noise Functions"""
    ###############################################

    # Add uniform noise to the image
    def addUniformNoise(self, a, b):
        if len(self.grayImage) != 0:
            self.clearImage()

            uniformNoise = np.random.uniform(a, b, self.grayImage.shape)
            uniformNoise = np.asarray(np.round(uniformNoise), dtype=np.int64)
            
            self.grayImage += uniformNoise
            self.grayImage = self.scaleImage(self.grayImage)
            # Draw image
            self.drawImage(self.grayImage)

    # Add gaussian noise to the image
    def addGaussianNoise(self, sigma, mean):
        if len(self.grayImage) != 0:
            self.clearImage()

            gaussianNoise = np.random.normal(mean, sigma, self.grayImage.shape)
            gaussianNoise = np.asarray(np.round(gaussianNoise), dtype=np.int64)
            
            self.grayImage += gaussianNoise
            self.grayImage = self.scaleImage(self.grayImage)

            # Draw image
            self.drawImage(self.grayImage)

    # Add rayleigh noise to the image
    def addRayleighNoise(self,scale):
        if len(self.grayImage) != 0:
            self.clearImage()

            rayleighNoise = np.random.rayleigh(scale, self.grayImage.shape)
            rayleighNoise = np.asarray(np.round(rayleighNoise), dtype=np.int64)
            
            self.grayImage += rayleighNoise
            self.grayImage = self.scaleImage(self.grayImage)
            
            # Draw image
            self.drawImage(self.grayImage)

    # TODO: Add erlang noise to the image
    def addErlangNoise(self, k, scale=1):
        if len(self.grayImage) != 0:
            self.clearImage()

            erlangNoise = np.random.gamma(k, scale, self.grayImage.shape)
            erlangNoise = np.asarray(np.round(erlangNoise), dtype=np.int64)
            
            self.grayImage += erlangNoise
            self.grayImage = self.scaleImage(self.grayImage)

            # Draw image
            self.drawImage(self.grayImage)

    # Add rayleigh noise to the image
    def addExponentialNoise(self,scale):
        if len(self.grayImage) != 0:
            self.clearImage()

            width, height = self.grayImage.shape
            exponentialNoise = np.random.exponential(scale, (width,height))
            exponentialNoise = np.asarray(np.round(exponentialNoise), dtype=np.int64)
            
            self.grayImage += exponentialNoise
            self.grayImage = self.scaleImage(self.grayImage)
            
            # Draw image
            self.drawImage(self.grayImage)

    # Add salt and pepper noise to the image
    def addSaltAndPepperNoise(self, mode="salt and pepper"):
        if len(self.grayImage) != 0:
            self.clearImage()
            width, height = self.grayImage.shape
            # Randomly pick some pixels in the image for coloring them white
            number_of_pixels = int((random.randint(2,7)/100) * (width*height))

            if mode == "salt and pepper":
                salt = True
                pepper = True
            elif mode == "salt":
                salt = True
                pepper = False
            elif mode == "pepper":
                salt = False
                pepper = True

            if pepper == True:
                for _ in range(number_of_pixels):        
                    self.grayImage[random.randint(0, width - 1)][random.randint(0, height - 1)] = 255

            if salt == True:
                for _ in range(number_of_pixels):        
                    self.grayImage[random.randint(0, width - 1)][random.randint(0, height - 1)] = 0
            
            # Draw image
            self.drawImage(self.grayImage)

    ###############################################
    """Operations Functions"""
    ###############################################

    def operationTwoImages(self, operation, image1, image2):
        if image1.shape != image2.shape:
            raise "Error"
            return

        if len(image1) == 0 or len(image2) == 0:
            return

        if operation == "subtract":
            resultedImage = image1 - image2
        elif operation == "add":
            resultedImage = np.add(image1, image2)
        elif operation == "multiply":
            resultedImage = np.multiply(image1, image2)
        elif operation == "divide":
            resultedImage = np.divide(image1, image2)

        elif operation == "union":
            resultedImage = np.union1d(image1, image2)
        elif operation == "intersect":
            resultedImage = np.intersect1d(image1, image2)
        
        elif operation == "and":
            resultedImage = np.bitwise_and(image1, image2)
        elif operation == "nand":
            resultedImage = np.bitwise_not(np.bitwise_and(image1, image2))
        elif operation == "or":
            resultedImage = np.bitwise_or(image1, image2)
        elif operation == "nor":
            resultedImage = np.bitwise_not(np.bitwise_or(image1, image2))
        elif operation == "xor":
            resultedImage = np.bitwise_xor(image1, image2)
        elif operation == "xnor":
            resultedImage = np.bitwise_not(np.bitwise_xor(image1, image2))

        # Scale
        scaledImage = self.scaleImage(resultedImage)
        self.grayImage = scaledImage

        # Draw image
        self.drawImage(self.grayImage)

    # Operation on one image
    def operationOneImages(self, operation):
        if len(self.grayImage) != 0:
            if operation == "not":
                resultedImage = np.bitwise_not(self.grayImage)
            elif operation == "complement":
                resultedImage = np.bitwise_not(self.grayImage)
            
            # Scale
            scaledImage = self.scaleImage(resultedImage)
            self.grayImage = scaledImage

            self.drawImage(self.grayImage, "Operation")

    ###############################################
    """Fourier Functions"""
    ###############################################
    
    # Fourier transform
    def fourierTransform(self, imageAtSpatialDomain, mode="fourier", log=False, draw=True, s=None):
        if len(imageAtSpatialDomain) != 0:
            if s == None:
                f = np.fft.fft2(imageAtSpatialDomain)
            else:
                f = np.fft.fft2(imageAtSpatialDomain, s)

            fshift = np.fft.fftshift(f)
            spectrum = None

            if mode == "magnitude":
                if not log:
                    spectrum = np.abs(fshift)
                else:
                    spectrum = logTransformation(np.abs(fshift))
            elif mode == "phase":
                spectrum  = np.angle(fshift)
            else:
                spectrum = f

            if draw:
                scaledSpectrum = self.scaleImage(spectrum)
                self.drawImage(scaledSpectrum, mode)

            return spectrum

    # inverse Fourier transform
    def inverseFourierTransform(self, combinedImage, mode="normal"):
        if mode=="separate":
            combinedImage = np.multiply(combinedImage[0], np.exp(1j * combinedImage[1]))

        # shiftedImage = np.fft.ifftshift(combinedImage)
        resultImage = np.fft.ifft2(combinedImage)
        return resultImage