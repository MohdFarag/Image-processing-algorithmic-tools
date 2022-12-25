# math & matrix computations library
import numpy as np
import random

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Phantom
from phantominator import shepp_logan
import skimage.transform # import iradon , radon, rotate, rescale

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
    def drawImage(self, image, title="Blank", cmap=plt.cm.Greys_r, scale="scale", save=True):
        self.clearImage()
        image = scaleImage(image, scale)

        if save:
            self.grayImage = image

        self.axes.set_title(title, fontsize = 16)
        self.axes.imshow(image, cmap=cmap, aspect='equal', origin='upper', vmin=0, vmax=255)
        self.draw()

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
        
        # To Draw ROI
        self.drawImage(ROI, scale="clip" ,save=False)
        
        # Get histogram of image
        L = ROI.max()
        histogram = np.bincount(ROI.flatten(), minlength=L)
        mean, variance, std = getStatisticsOfHistogram(histogram, L)

        # self.grayImage = ROI
        return ROI, mean, variance, std

    ###############################################
    """Image Functions"""
    ###############################################

    # Draw Binary image
    def binaryImage(self):
        self.grayImage = binaryImage(self.grayImage)
        self.drawImage(self.grayImage)

    # Draw logged image
    def logImage(self):
        self.grayImage = logTransformation(self.grayImage)
        self.drawImage(self.grayImage)

    # Apply negative on image
    def negativeImage(self):
        self.grayImage = negativeImage(self.grayImage)
        self.drawImage(self.grayImage)

    # Apply gamma correction on image
    def gammaCorrectionImage(self, Y):
        self.grayImage = gammaCorrectionImage(self.grayImage, Y)
        self.drawImage(self.grayImage)

    ###############################################
    """Piecewise-Linear Intensity Transformation Functions"""
    ###############################################

    def contrastStretching(self, r1, s1, r2, s2):
        self.grayImage = contrastStretching(self.grayImage, r1, s1, r2, s2)
        self.drawImage(self.grayImage)

    def intensityLevelSlicing(self, A, B, mode):
        self.grayImage = intensityLevelSlicing(self.grayImage, A, B, mode)
        self.drawImage(self.grayImage)

    def bitPlaneSlicing(self):
        if self.loaded:
            fig, ax = plt.subplots(2,4)
            i, j = 0,0
            for bit in range(1,9):
                slicingImage = bitPlaneSlicing(self.grayImage, 1, bit)
                ax[i,j].imshow(slicingImage, cmap="gray")
                ax[i,j].set_title(f'{bit} bit')
                if j != 3:
                    j += 1
                else:
                    i += 1
                    j = 0

            fig.show()

    ###############################################
    """Transformations Functions"""
    ###############################################

    # Zoom image
    def zoomImage(self, scaleFactor, mode):
        if self.loaded and self.grayImage.ndim == 2:
            resizedImage = zoom(self.grayImage, scaleFactor, mode)

            zoomWidth = int(resizedImage.shape[0]/scaleFactor)
            zoomHeight = int(resizedImage.shape[1]/scaleFactor)
            
            self.grayImage = resizedImage[:zoomWidth,:zoomHeight]
            self.drawImage(self.grayImage)
            return resizedImage.shape[0], resizedImage.shape[1]
        else:
            return "N/A","N/A"

    # Rotate image
    def rotateImage(self, image, angle, mode="nearest"):
        if len(image) != 0:
            self.grayImage = rotate(self.grayImage, angle, mode)
            self.drawImage(self.grayImage)
            return self.grayImage.shape[0], self.grayImage.shape[1]   
        else:
            return "N/A","N/A"
             
    # Shear image
    def shearImage(self, angle, mode="horizontal"):
        if self.loaded:
            self.grayImage = shear(self.grayImage, angle, mode)
            self.drawImage(self.grayImage)
        else:
            return

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
    def constructTriangle(self):
        self.grayImage = np.zeros((128,128), dtype=np.int64)
        
        k = 100 - 29
        for i in range(29,100):
            for j in range(k,127-k):
                self.grayImage[i,j] = 255
            k -= 1

        self.loaded = True
        self.drawImage(self.grayImage)

    # Construct Circle in gray box shape
    def constructCircle(self, n=256, m=256, I1=250, I2=150, I3=50, d=184, r=64):
        self.grayImage = np.ones((n,m), dtype=np.int64)
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
        self.drawImage(self.grayImage, scale="clip")

    # Construct Square shape
    def constructSquare(self):
        # Parameters
        n,m = 128,128
        length = 64

        self.grayImage = np.zeros((n,m), dtype=np.int64)
        differenceSize = (n - length) // 2

        for i in range(n):
            for j in range(m):
                if (differenceSize <= i <= n-differenceSize) and (differenceSize <= j <= m-differenceSize):
                    self.grayImage[i,j] = 255
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

    ###############################################
    """Phantom"""
    ###############################################

    # Construct phantom
    def constructPhantom(self, size=256):
        self.loaded = True
        sheppLoganPhantom = np.flip(shepp_logan(size))
        self.drawImage(sheppLoganPhantom)
        
    # Display a Sinogram of this phantom
    def drawSinogram(self, image, angles=np.arange(180)):
        if len(image) != 0:
            # Get sinogram
            self.grayImage = radon(image, angles)
            self.drawImage(self.grayImage, "Sinogram")
            self.grayImage = np.rot90(self.grayImage,3)

    # Display a Laminogram of this phantom from sinogram
    def drawLaminogram(self, sinogram, thetas=range(180), filterType=None):
        if len(sinogram) != 0:
            laminogram = skimage.transform.iradon(sinogram[:,thetas], thetas, filter_name=filterType)
            self.drawImage(laminogram, title="Laminogram")

    ###############################################
    """Histogram Functions"""
    ###############################################

    # Get histogram image
    def drawHistogram(self, image:np.ndarray):
        if len(image) != 0:     
            self.clearImage()    
            L = image.max()
            # Get histogram of image
            histogram = getHistogram(image, L)

            # Calculate statistics of histogram
            mean, variance, std = getStatisticsOfHistogram(histogram, L)
            
            # Normalize
            sumPixels = np.sum(histogram)
            normalizedHistogram = histogram/sumPixels

            cdf = normalizedHistogram.cumsum()
            cdf_normalized = cdf * float(normalizedHistogram.max()) / cdf.max()

            self.axes.bar(range(len(normalizedHistogram)), normalizedHistogram, color='red')
            self.axes.plot(cdf_normalized, color = 'black')
            self.axes.axvline(mean, color='g', linestyle='dashed', linewidth=1)
            self.axes.legend(('cumulative histogram','mean','histogram'), loc = 'upper left')
           
            self.axes.set_title(f'\u03BC = {mean}    \u03C3 = {std}')
            self.draw()
        else:
            return

    # Normalized Histogram
    def drawNormalizedHistogram(self):
        if len(self.grayImage) != 0:
            self.grayImage = normalizeHistogram(self.grayImage)            
            # Draw equalized image
            self.drawImage(self.grayImage)
        else:
            return

    ###############################################
    """Filters Functions"""
    ###############################################

    # Apply specific 'spatial' filter
    # You can choose whether the filter applied in spatial or frequency domain
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
                blurredImage = convolution(self.grayImage, kernel)
            
            elif domain == "frequency": 
                # Image fourier
                image = addPadding(image, (xImagePadding,yImagePadding))               
                grayImageInFreqDomain = self.fourierTransform(image, draw=False, s=size)
                
                # Kernel fourier
                kernel = addPadding(kernel, (xPaddingFilterSize,yPaddingFilterSize))
                boxFilterInFreqDomain = self.fourierTransform(kernel, draw=False,s=size)

                filteredImageInFreqDomain = boxFilterInFreqDomain * grayImageInFreqDomain

                blurredImage = self.inverseFourierTransform(filteredImageInFreqDomain)
                blurredImage = np.fft.fftshift(blurredImage)

                blurredImage = np.abs(blurredImage)
                blurredImage = blurredImage[xImagePadding:rows-xImagePadding,yImagePadding:cols-yImagePadding]
                
            return blurredImage

    # Subtract blurred image from original
    def subtractBlurredFromOriginal(self, blurredImage):
        resultImage = np.subtract(self.grayImage, blurredImage)
        return resultImage
    
    # Multiply by a factor K Then added to the original image
    def multiplyByFactor(self, image, k):
        resultImage = np.multiply(image,k)
        resultImage = np.add(resultImage, self.grayImage)

        return resultImage
    
    # Order statistic mask
    def OrderStatisticFilter(self, size, percent):
        if len(self.grayImage) != 0:
            self.grayImage = OrderStatisticFilter(self.grayImage, size, percent)
            # Draw image
            self.drawImage(self.grayImage)

    # Perform un-sharp masking
    def unsharpMask(self, size, k):
        if len(self.grayImage) != 0:
            boxFilter = boxKernel(size)
            # Apply box kernel
            blurredImage = self.applySpatialFilter(self.grayImage, boxFilter)
            # Subtract blurred image from original image
            subtractedImage = self.subtractBlurredFromOriginal(blurredImage)
            # Multiply the result by k (highboost factor) then sum to original image
            self.grayImage = self.multiplyByFactor(subtractedImage, k)          
            # Draw image
            self.drawImage(self.grayImage)
    
    # Perform box filtering
    def boxFiltering(self, size):
        if len(self.grayImage) != 0:
            boxFilter = boxKernel(size)
            self.grayImage = self.applySpatialFilter(self.grayImage, boxFilter)
                       
            # Draw image
            self.drawImage(self.grayImage)

    # Perform box filtering in frequency domain
    def boxFilteringUsingFourier(self, filterSize):
        if len(self.grayImage) != 0:
            boxFilter = boxKernel(filterSize)

            self.grayImage = self.applySpatialFilter(self.grayImage, boxFilter, domain="frequency")
            
            # Draw image
            self.drawImage(self.grayImage)

    # Notch reject filter
    def notchRejectFilter(self, magnitudeSpectrum, points, d0=9):
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
        
            self.grayImage = np.abs(self.inverseFourierTransform(magnitudeSpectrum))
            # Draw image
            self.drawImage(self.grayImage)
            
    ###############################################
    """Noise Functions"""
    ###############################################

    # Add uniform noise to the image
    def addUniformNoise(self, a, b):
        if len(self.grayImage) != 0:
            uniformNoise = np.random.uniform(a, b, self.grayImage.shape)
            uniformNoise = np.asarray(np.round(uniformNoise), dtype=np.int64)
            self.grayImage += uniformNoise

            # Draw image
            self.drawImage(self.grayImage, scale="clip")

    # Add gaussian noise to the image
    def addGaussianNoise(self, mean, sigma):
        if len(self.grayImage) != 0:
            gaussianNoise = np.random.normal(mean, sigma, self.grayImage.shape)
            gaussianNoise = np.asarray(np.round(gaussianNoise), dtype=np.int64)            
            self.grayImage += gaussianNoise

            # Draw image
            self.drawImage(self.grayImage, scale="clip")

    # Add rayleigh noise to the image
    def addRayleighNoise(self, mode):
        if len(self.grayImage) != 0:
            rayleighNoise = np.random.rayleigh(mode, self.grayImage.shape)
            rayleighNoise = np.asarray(np.round(rayleighNoise), dtype=np.int64)            
            self.grayImage += rayleighNoise
            
            # Draw image
            self.drawImage(self.grayImage)

    # Add erlang noise to the image
    def addErlangNoise(self, k, scale=1):
        if len(self.grayImage) != 0:
            erlangNoise = np.random.gamma(k, scale, self.grayImage.shape)
            erlangNoise = np.asarray(np.round(erlangNoise), dtype=np.int64)            
            self.grayImage += erlangNoise

            # Draw image
            self.drawImage(self.grayImage)

    # Add rayleigh noise to the image
    def addExponentialNoise(self, scale):
        if len(self.grayImage) != 0:
            width, height = self.grayImage.shape
            exponentialNoise = np.random.exponential(scale, (width,height))
            exponentialNoise = np.asarray(np.round(exponentialNoise), dtype=np.int64)            
            self.grayImage += exponentialNoise
            
            # Draw image
            self.drawImage(self.grayImage)

    # Add salt & pepper noise to the image
    def addSaltAndPepperNoise(self, mode="salt and pepper"):
        if len(self.grayImage) != 0:
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

    # Operation on two images
    def operationTwoImages(self, operation, image1, image2):
        # if image1.shape != image2.shape:
        if len(image1) == 0 or len(image2) == 0:
            return

        rowPadding1, colPadding1 = (image2.shape[0] - image1.shape[0]) // 2, (image2.shape[1] - image1.shape[1]) // 2
        rowPadding2, colPadding2 = (image1.shape[0] - image2.shape[0]) // 2, (image1.shape[1] - image2.shape[1]) // 2

        image1 = addPadding(image1,(rowPadding1, colPadding1))
        image2 = addPadding(image2,(rowPadding2, colPadding2))

        if operation == "subtract":
            resultedImage = np.subtract(image1, image2)
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

        # Draw image
        self.drawImage(resultedImage)

    # Operation on one image
    def operationOneImages(self, operation):
        if len(self.grayImage) != 0:
            if operation == "not":
                resultedImage = np.bitwise_not(self.grayImage)
            elif operation == "complement":
                resultedImage = np.bitwise_not(self.grayImage)
            
            self.drawImage(resultedImage, "Operation")

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
                spectrum = f

            if draw:
                self.drawImage(spectrum, mode)

            return spectrum

    # inverse Fourier transform
    def inverseFourierTransform(self, combinedImage, mode="normal"):
        if mode=="separate":
            combinedImage = np.multiply(combinedImage[0], np.exp(1j * combinedImage[1]))

        # shiftedImage = np.fft.ifftshift(combinedImage)
        resultImage = np.fft.ifft2(combinedImage)
        return resultImage

    ###############################################
    """Morphological Functions"""
    ###############################################
    
    # inverse Fourier transform
    def morphologicalActions(self, option):
        if len(self.grayImage) != 0:
            image = binaryImage(self.grayImage)
            if option == 'erosion':
                print(1)
                result = erosionImage(image)
            elif option == 'dilation':
                print(2)
                result = dilationImage(image)
            elif option == 'opening':
                print(3)
                result = opening(image)
            elif option == 'closing':
                print(4)
                result = closing(image)
            elif option == 'noise':
                print(5)
                result = removeNoise(image)

            self.drawImage(result)