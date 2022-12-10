# math & matrix computations library
import math
import numpy as np
import random

# Matplotlib
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.figure import Figure 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    # Set Image
    def setImage(self, image_path, fileExtension):
        # Reading the image
        if fileExtension == "dcm":
            imgData = dicom.dcmread(image_path, force=True)
            downloadedImage = imgData.pixel_array
        else:
            downloadedImage = mpimg.imread(image_path)
            imgData = Image.open(image_path)

        # If image is RGB transform it to gray.
        if downloadedImage.ndim > 2:
            self.grayImage = downloadedImage[:,:,0]
        else:
            self.grayImage = downloadedImage

        self.grayImage = self.scaleImage(self.grayImage)

        self.loaded = True
        self.axes.imshow(self.grayImage, cmap="gray")
        self.draw()

        # Depends on extension of the file
        return imgData

    # Scale function
    def scaleImage(self, image, mode="scale", a_min=0, a_max=255):
        resultImage = np.zeros(image.shape)        
        
        if mode == "scale":
            image = image - image.min()
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
    
    # Log transformation
    def logTransformation(self, r:np.ndarray):
        maxPixelValue = np.max(r)
        c = 255 / (np.log(1 + maxPixelValue))        
        result = np.log(1 + r)

        return result

    ###############################################
    """Edit Functions"""
    ###############################################

    # Draw logged image
    def logImage(self):
        self.grayImage = self.logTransformation(self.grayImage)
        self.axes.imshow(self.grayImage, cmap="gray")
        self.draw() 

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
            # Initilize resized image
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
        
        self.grayImage = resizedImage[:zoomWidth,:zoomHeight]
        self.axes.imshow(self.grayImage, cmap="gray")
        self.draw()

        return resizedImage.shape[0], resizedImage.shape[1]

    # Rotate image
    def rotateImage(self, angle, mode):
        if self.loaded:
            # Converting degrees to radians
            angle = math.radians(angle)

            cosine = math.cos(angle)
            sine = math.sin(angle)

            # Define the height of the image
            oldWidth = self.grayImage.shape[0]
            # Define the width of the image
            oldHeight = self.grayImage.shape[1]

            sizeImage = min(oldWidth,oldHeight)
          
            # Initilize rotated image 
            rotatedImage = np.zeros((sizeImage,sizeImage)) 

            # Find the center of the Rotated T image
            centerHeight = int( (sizeImage+1)/2) # mid row
            centerWidth = int( (sizeImage+1)/2 ) # mid col

            for i in range(sizeImage):
                for j in range(sizeImage):
                    x = -(j-centerWidth)*sine + (i-centerHeight)*cosine
                    y = (j-centerWidth)*cosine + (i-centerHeight)*sine
                    
                    # Add offset
                    x += centerHeight
                    y += centerWidth
                    
                    if mode == "nearest":
                        # Get nearest index
                        x = round(x)
                        y = round(y)
                        #  check if x/y corresponds to a valid pixel in input image
                        if (x >= 0 and y >= 0 and x < sizeImage and y < sizeImage):
                            rotatedImage[j][i] = self.grayImage[y][x]
        
                    elif mode == "linear":    
                        # Calculate the coordinate values for 4 surrounding pixels.
                        x_floor = math.floor(x)
                        x_ceil = min(oldWidth-1, math.ceil(x))
                        y_floor = math.floor(y)
                        y_ceil = min(oldHeight - 1, math.ceil(y))
                        
                        if (x >= 0 and y >= 0 and x < sizeImage and y < sizeImage):
                            if (x_ceil == x_floor) and (y_ceil == y_floor):
                                q = self.grayImage[int(y), int(x)]
                            elif (y_ceil == y_floor):
                                q1 = self.grayImage[int(y), int(x_floor)]
                                q2 = self.grayImage[int(y), int(x_ceil)]
                                q = q1 * (x_ceil - x) + q2 * (x - x_floor)
                            elif (x_ceil == x_floor):
                                q1 = self.grayImage[int(y_floor), int(x)]
                                q2 = self.grayImage[int(y_ceil), int(x)]
                                q = (q1 * (y_ceil - y)) + (q2 * (y - y_floor))
                            else:
                                p1 = self.grayImage[y_floor, x_floor]
                                p2 = self.grayImage[y_ceil, x_floor]
                                p3 = self.grayImage[y_floor, x_ceil]
                                p4 = self.grayImage[y_ceil, x_ceil]

                                q1 = p1 * (y_ceil - y) + p2 * (y - y_floor)
                                q2 = p3 * (y_ceil - y) + p4 * (y - y_floor)
                                q = q1 * (x_ceil - x) + q2 * (x - x_floor)

                            rotatedImage[j][i] = q
            
            self.grayImage = rotatedImage
            self.axes.imshow(self.grayImage, cmap="gray")
            self.draw()

            return rotatedImage.shape[0],rotatedImage.shape[1]
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
                    elif mode == "vertical":
                        new_x = x
                        new_y = round(x-y*tangent)
                    else:
                        new_x = x
                        new_y = round(x-y*tangent)

                    # Add offset
                    new_x += centerHeight
                    new_y += centerWidth

                    if (new_x >= 0 and new_y >= 0 and new_x < oldWidth and new_y < oldHeight):
                        shearedImage[j][i] = self.grayImage[new_y,new_x]

            self.grayImage = shearedImage
            self.axes.imshow(self.grayImage, cmap="gray")
            self.draw()

    ###############################################
    """Shapes construction Functions"""
    ###############################################

    # Construct T shape
    def constructT(self, background="white"):
        self.grayImage = np.zeros((128,128), dtype=np.int64)
        
        if background == "black":
            self.grayImage.fill(255)

        for i in range(29,50):
            for j in range(29,100):
                self.grayImage[i,j] = 255
        
        for i in range(49,100):
            for j in range(54,74):
                self.grayImage[i,j] = 255

        self.axes.imshow(self.grayImage, cmap="gray")
        self.loaded = True
        self.draw()

    # Construct Triangle shape
    def constructTriangle(self, background="white"):
        self.grayImage = np.zeros((128,128), dtype=np.int64)
        
        if background == "black":
            self.grayImage.fill(255)

        k = 100 - 29
        for i in range(29,100):
            for j in range(k,127-k):
                self.grayImage[i,j] = 255
            k -= 1

        self.axes.imshow(self.grayImage, cmap="gray")
        self.loaded = True
        self.draw()

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
            L = 256
          
            # Get histogram of image
            try:
                histogram = np.bincount(np.asarray(np.round(image),np.int64).flatten(), minlength=L)
            except:
                histogram = np.bincount(np.arange(L).flatten(), minlength=L)

            # Normalize
            sumPixels = np.sum(histogram)
            normalizedHistogram = histogram/sumPixels

            cdf = normalizedHistogram.cumsum()
            cdf_normalized = cdf * float(normalizedHistogram.max()) / cdf.max()

            self.axes.bar(range(len(normalizedHistogram)), normalizedHistogram, color='red')
            self.axes.plot(cdf_normalized, color = 'black')
            
            self.axes.legend(('cumulative histogram','histogram'), loc = 'upper left')

            self.draw()
        else:
            return

    # Normalized Histogram
    def normalizeHistogram(self):
        if len(self.grayImage) != 0:
            self.clearImage() # Clear prev.
            L = self.grayImage.max()
          
            # Get histogram of nonEqualizedImage
            nonEqualizedhistogram = np.bincount(self.grayImage.flatten(), minlength=L)

            # Normalize
            sumPixels = np.sum(nonEqualizedhistogram)
            nonEqualizedhistogram = nonEqualizedhistogram/sumPixels

            # Normalized cumulative histogram
            cfdHistogram = np.cumsum(nonEqualizedhistogram)

            # Initilized transform map
            transformMap = np.floor((L-1) * cfdHistogram)

            # Flatten image array into 1D list
            flatNonEqualizedImage = list(self.grayImage.flatten())
            flatEqualizedImage = [transformMap[p] for p in flatNonEqualizedImage]

            # Reshape and write back into equalizedImage
            self.grayImage = np.reshape(np.asarray(flatEqualizedImage, dtype=np.int64), self.grayImage.shape)

            # Draw equalized image
            self.axes.imshow(self.grayImage, cmap="gray")
            self.draw()
        else:
            return

    ###############################################
    """Filters Functions"""
    ###############################################

    # Add padding to image
    def addPadding(self, image, paddingSize, mode="same", value=0):
        addedPadding = 2 * paddingSize
        resultImage = np.zeros((image.shape[0] + addedPadding,image.shape[1] + addedPadding))
        
        if mode == "same":
            resultImage.fill(value)
            for i in range(paddingSize, resultImage.shape[0] - paddingSize):
                for j in range(paddingSize, resultImage.shape[1] - paddingSize):
                    resultImage[i][j] = image[i-paddingSize][j-paddingSize] 
        
        return resultImage

    # Convolution function
    def convolution(self, filter:np.ndarray, mode="convolution"):
        if mode == "convolution":
            filter = np.flip(filter)

        filterWidth = filter.shape[0]
        filterHeight = filter.shape[1]
       
        filterSize = max(filterWidth, filterHeight)
        paddingSize = filterSize // 2

        paddedImage = self.addPadding(self.grayImage, paddingSize)

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

        filter = np.zeros(shape)
        filter.fill(1/(size*size))

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
            self.axes.imshow(self.grayImage, cmap="gray")
            self.draw() 

    # Perform un-sharp masking
    def unsharpMask(self, size, k):
        if len(self.grayImage) != 0:
            self.clearImage()
            blurredImage = self.boxFiltering(size, True)

            subtractedImage = self.subtractBlurredFromOriginal(blurredImage)
            resultImage = self.multiplyByFactor(subtractedImage, k)

            scaledImage = self.scaleImage(resultImage)
            self.grayImage = scaledImage
            
            # Draw image
            self.axes.imshow(self.grayImage, cmap="gray")
            self.draw()
    
    # Perform box filtering
    def boxFiltering(self, size, output=False):
        if len(self.grayImage) != 0:
            self.clearImage()

            boxFilter = self.boxKernel(size)
            blurredImage = self.convolution(boxFilter)
            
            if output:
                return blurredImage
            else:
                # Scale
                scaledImage = self.scaleImage(blurredImage)
                self.grayImage = scaledImage
                # Draw image
                self.axes.imshow(self.grayImage, cmap="gray")
                self.draw()

    # Perform box filtering in frequency domain
    def boxFilteringUsingFourier(self, filterSize):
        if len(self.grayImage) != 0:
            self.clearImage()

            paddingSize = filterSize // 2

            fftRows = self.grayImage.shape[0] + filterSize - 1
            fftCols = self.grayImage.shape[1] + filterSize - 1
            fftShape = (fftRows, fftCols)

            boxFilter = self.boxKernel(filterSize)

            boxFilterInFreqDomain = np.fft.fft2(boxFilter, fftShape)
            grayImageInFreqDomain = np.fft.fft2(self.grayImage, fftShape)

            filteredImageInFreqDomain = boxFilterInFreqDomain * grayImageInFreqDomain

            invertedImage = self.inverseFourierTransform(filteredImageInFreqDomain)
            resultImage = np.abs(invertedImage)
            
            resultImage = resultImage[paddingSize:fftRows-paddingSize,paddingSize:fftCols-paddingSize]
            scaledImage = self.scaleImage(resultImage)
            self.grayImage = scaledImage

            # Draw image
            self.axes.imshow(self.grayImage, cmap="gray")
            self.draw()

    ###############################################
    """Noise Functions"""
    ###############################################

    # Add salt and pepper noise to the image
    def addSaltAndPepper(self):
        if len(self.grayImage) != 0:
            self.clearImage()

            width, height = self.grayImage.shape

            # Randomly pick some pixels in the image for coloring them white
            number_of_pixels = int((random.randint(2,7)/100) * (width*height))
            for _ in range(number_of_pixels):        
                self.grayImage[random.randint(0, width - 1)][random.randint(0, height - 1)] = 255
                
            for _ in range(number_of_pixels):        
                self.grayImage[random.randint(0, width - 1)][random.randint(0, height - 1)] = 0
            
            # Draw image
            self.axes.imshow(self.grayImage, cmap="gray")
            self.draw() 

    ###############################################
    """Operations Functions"""
    ###############################################

    def subtractionTwoImage(self, image1, image2):
        if len(image1) != 0 and len(image2) != 0:
            subtractedImage = np.subtract(image1, image2)
            # Scale
            scaledImage = self.scaleImage(subtractedImage)
            self.grayImage = scaledImage

            # Draw image
            self.axes.imshow(self.grayImage, cmap="gray")
            self.draw()
        
        return

    def additionTwoImage(self, image1, image2):
        if len(image1) != 0 and len(image2) != 0:
            subtractedImage = np.add(image1, image2)
            # Scale
            scaledImage = self.scaleImage(subtractedImage)
            self.grayImage = scaledImage

            # Draw image
            self.axes.imshow(self.grayImage, cmap="gray")
            self.draw()
        
        return

    ###############################################
    """Fourier Functions"""
    ###############################################
    
    # Fourier transform
    def fourierTransform(self, imageAtSpatialDomain, mode="fourier", log=False, draw=True, s=(3,3)):
        if len(imageAtSpatialDomain) != 0:
            f = np.fft.fft2(imageAtSpatialDomain)
            fshift = np.fft.fftshift(f)
            spectrum = None

            if mode == "magnitude":
                if not log:
                    spectrum = np.abs(fshift)
                else:
                    spectrum = self.logTransformation(np.abs(fshift))
            elif mode == "phase":
                spectrum  = np.angle(fshift)
            else:
                spectrum = f

            if draw:
                scaledSpectrum = self.scaleImage(spectrum)
                self.axes.imshow(scaledSpectrum, cmap="gray")
                self.draw() 

            return spectrum

    # inverse Fourier transform
    def inverseFourierTransform(self, combinedImage, mode="normal"):
        if mode=="separate":
            combinedImage = np.multiply(combinedImage[0], np.exp(1j * combinedImage[1]))

        resultImage = np.fft.ifft2(combinedImage)
        return resultImage