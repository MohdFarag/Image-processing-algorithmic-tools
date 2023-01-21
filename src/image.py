""" Image Viewer Class"""

# pylint: disable=C0103, W0105, C0301, W0613, E1136

# math & matrix computations library
import random
import numpy as np

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

# Pydicom
import pydicom as dicom

from PIL import Image
from .utilities import *

class ImageViewer(FigureCanvasQTAgg):
    """Image Viewer Class

    Args:
        FigureCanvasQTAgg (_type_)
    """

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

        plt.style.context('fivethirtyeight')

    ###############################################
    """Image Functions"""
    ###############################################

    # Set image
    def setImage(self, image_path, fileExtension):
        # Reading the image
        if fileExtension == "dcm":
            imgInformation = dicom.dcmread(image_path, force=True)
            self.originalImage = imgInformation.pixel_array
        else:
            self.originalImage = mpimg.imread(image_path)
            imgInformation = Image.open(image_path)

        # If image is RGB transform it to gray.
        if self.originalImage.ndim > 2:
            self.grayImage = self.originalImage[:,:,0]
        else:
            self.grayImage = self.originalImage

        self.loaded = True
        self.drawImage(self.grayImage)

        # Depends on extension of the file
        return imgInformation

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
        corners = np.int32(np.round(corners))
        p1 = corners[0][0], corners[1][0]
        p2 = corners[0][2], corners[1][2]
        ROI = self.grayImage[p1[1]:p2[1], p1[0]:p2[0]]

        # To Draw ROI
        self.drawImage(ROI, scale="clip" ,save=False)
        return ROI

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

    # Scale image
    def scaleImage(self, scaleFactor, mode):
        if self.loaded and self.grayImage.ndim == 2:
            resizedImage = scale(self.grayImage, scaleFactor, mode)

            scaleWidth = int(resizedImage.shape[0]/scaleFactor)
            scaleHeight = int(resizedImage.shape[1]/scaleFactor)

            self.grayImage = resizedImage[:scaleWidth,:scaleHeight]
            self.drawImage(self.grayImage)

    # Rotate image
    def rotateImage(self, angle, mode="nearest"):
        if len(self.grayImage) != 0:
            self.grayImage = rotate(self.grayImage, angle, mode)
            self.drawImage(self.grayImage) 

    # Shear image
    def shearImage(self, angle, mode="horizontal"):
        if self.loaded:
            self.grayImage = shear(self.grayImage, angle, mode)
            self.drawImage(self.grayImage)
        else:
            return

    # Shear image
    def translateImage(self, x, y):
        if self.loaded:
            self.grayImage = translate(self.grayImage, x, y)
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

            self.axes.set_title(f'\u03BC = {mean:.2f}    \u03C3\u00b2 = {variance:.2f}    \u03C3 = {std:.2f}')
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

    # Order statistic mask
    def OrderStatisticFilter(self, size, percent):
        if len(self.grayImage) != 0:
            self.grayImage = orderStatisticFilter(self.grayImage, size, percent)
            # Draw image
            self.drawImage(self.grayImage)

    # Perform un-sharp masking
    def unsharpMask(self, size, k, domain="spatial", filterType="lowpass"):
        if len(self.grayImage) != 0:
            self.grayImage = unsharpMask(self.grayImage, size, k, domain=domain, filterType=filterType)
            # Draw image
            self.drawImage(self.grayImage)


    def homomorphicFilter(self, d0, c, gammaL, gammaH):
        if len(self.grayImage) != 0:
            kernel = homomorphicFilter(self.grayImage.shape, d0, c, gammaL, gammaH)
            self.grayImage = applyFrequencyFilter(self.grayImage, kernel, domain="spatial")
            
            # Draw image
            self.drawImage(self.grayImage)
            
    ### low pass filters ###
    
    # Perform box filtering in spatial domain or frequency domain
    def boxFiltering(self, size, domain="spatial"):
        if len(self.grayImage) != 0:
            boxFilter = boxFilter(size)
            self.grayImage = applySpatialFilter(self.grayImage, boxFilter, domain=domain)
            
            # Draw image
            self.drawImage(self.grayImage)

    # Perform Ideal low pass filter by applying a mask in frequency domain
    def idealLowPassFilter(self, d0=9):
        if len(self.grayImage) != 0:
            kernel = idealLowPassFilter(self.grayImage.shape, d0)
            self.grayImage = applyFrequencyFilter(self.grayImage, kernel, domain="spatial")
            
            # Draw image
            self.drawImage(self.grayImage)
    
    # Perform gaussian filtering
    def gaussianFiltering(self, sigma, size, domain="spatial"):
        if len(self.grayImage) != 0:
            gaussianFilter = gaussianFilter(sigma,size)
            self.grayImage = applySpatialFilter(self.grayImage, gaussianFilter, domain=domain)

            # Draw image
            self.drawImage(self.grayImage)
            
    # Perform Gaussian low pass filter by applying a mask in frequency domain
    def gaussianLowPassFilter(self, d0=9):
        if len(self.grayImage) != 0:
            kernel = gaussianLowPassFilter(self.grayImage.shape, d0)
            self.grayImage = applyFrequencyFilter(self.grayImage, kernel, domain="spatial")
            
            # Draw image
            self.drawImage(self.grayImage)

    # Perform Butterworth low pass filter by applying a mask in frequency domain
    def butterworthLowPassFilter(self, d0=9, n=2):
        if len(self.grayImage) != 0:
            kernel = butterworthLowPassFilter(self.grayImage.shape, d0, n)
            self.grayImage = applyFrequencyFilter(self.grayImage, kernel, domain="spatial")
            
            # Draw image
            self.drawImage(self.grayImage)
    
    ### High pass filters ###
    
    # Perform Laplacian filter
    def laplacianFiltering(self, domain="spatial", size=3):
        if len(self.grayImage) != 0:
            if domain == "frequency":
                kernel = laplacianFilter(shape=self.grayImage.shape)
                self.grayImage = applyFrequencyFilter(self.grayImage, kernel)
            else:
                kernel = laplacianFilter(domain='spatial', shape=(size,size))
                self.grayImage = applySpatialFilter(self.grayImage, kernel)
            
            # Draw image
            self.drawImage(self.grayImage)

    # Perform Ideal high pass filter by applying a mask in frequency domain
    def idealHighPassFilter(self, d0=9):
        if len(self.grayImage) != 0:
            kernel = idealHighPassFilter(self.grayImage.shape, d0)
            self.grayImage = applyFrequencyFilter(self.grayImage, kernel, domain="spatial")
            
            # Draw image
            self.drawImage(self.grayImage)
            
    # Perform Gaussian high pass filter by applying a mask in frequency domain
    def gaussianHighPassFilter(self, d0=9):
        if len(self.grayImage) != 0:
            kernel = gaussianHighPassFilter(self.grayImage.shape, d0)
            self.grayImage = applyFrequencyFilter(self.grayImage, kernel, domain="spatial")
            
            # Draw image
            self.drawImage(self.grayImage)

    # Perform Butterworth high pass filter by applying a mask in frequency domain
    def butterworthHighPassFilter(self, d0=9, n=2):
        if len(self.grayImage) != 0:
            kernel = butterworthHighPassFilter(self.grayImage.shape, d0, n)
            self.grayImage = applyFrequencyFilter(self.grayImage, kernel, domain="spatial")
            
            # Draw image
            self.drawImage(self.grayImage)
    
    ### Band pass filters ###
    
    # Band reject filter
    def bandRejectFilter(self, magnitudeSpectrum, points, d0=9):
        if len(self.grayImage) != 0:       
            resultImage = bandRejectFilter(self.grayImage,magnitudeSpectrum,points,d0)

            # Draw image
            self.drawImage(resultImage)

    ###############################################
    """Noise Functions"""
    ###############################################

    # Add periodic noise to the image
    def addPeriodicNoise(self, amplitude, frequency, phase, center=0):
        if len(self.grayImage) != 0:
            x = np.linspace(0,1 ,self.grayImage.shape[1])
            y = np.linspace(0,1 ,self.grayImage.shape[0])
            x, y = np.meshgrid(x, y)

            periodicNoise = sinusoidal(x+y,amplitude, frequency, phase, center)
            self.grayImage = addNoise(self.grayImage, periodicNoise, scale=True)

            # Draw image
            self.drawImage(self.grayImage)
    
    # Add uniform noise to the image
    def addUniformNoise(self, a, b):
        if len(self.grayImage) != 0:
            uniformNoise = np.random.uniform(a, b, self.grayImage.shape)
            self.grayImage = addNoise(self.grayImage, uniformNoise)

            # Draw image
            self.drawImage(self.grayImage)

    # Add gaussian noise to the image
    def addGaussianNoise(self, mean, sigma):
        if len(self.grayImage) != 0:
            gaussianNoise = np.random.normal(mean, sigma, self.grayImage.shape)
            self.grayImage = addNoise(self.grayImage, gaussianNoise)


            # Draw image
            self.drawImage(self.grayImage)

    # Add rayleigh noise to the image
    def addRayleighNoise(self, mode):
        if len(self.grayImage) != 0:
            rayleighNoise = np.random.rayleigh(mode, self.grayImage.shape)
            self.grayImage = addNoise(self.grayImage, rayleighNoise)

            # Draw image
            self.drawImage(self.grayImage)

    # Add erlang noise to the image
    def addErlangNoise(self, k, scale=1):
        if len(self.grayImage) != 0:
            erlangNoise = np.random.gamma(k, scale, self.grayImage.shape)
            self.grayImage = addNoise(self.grayImage, erlangNoise)

            # Draw image
            self.drawImage(self.grayImage)

    # Add rayleigh noise to the image
    def addExponentialNoise(self, scale):
        if len(self.grayImage) != 0:
            exponentialNoise = np.random.exponential(scale, self.grayImage.shape)
            self.grayImage = addNoise(self.grayImage, exponentialNoise)

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

            if pepper is True:
                for _ in range(number_of_pixels):        
                    self.grayImage[random.randint(0, width - 1)][random.randint(0, height - 1)] = 255

            if salt is True:
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
            resultedImage = union2d(image1, image2)
        elif operation == "intersect":
            resultedImage = intersect2d(image1, image2)
        
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
        self.drawImage(resultedImage, scale="clip")

    # Operation on one image
    def operationOneImages(self, operation):
        if len(self.grayImage) != 0:
            if operation == "not":
                resultedImage = np.bitwise_not(self.grayImage)
            elif operation == "complement":
                resultedImage = complement2d(self.grayImage)

            self.drawImage(resultedImage, "Operation")

    ###############################################
    """Fourier Functions"""
    ###############################################

    # Fourier transform
    def fourierTransform(self, image, mode="fourier", log=False):
        if len(image) != 0:
            spectrum = fourierTransform(image,mode,log)            
            self.drawImage(spectrum, mode)

    ###############################################
    """Morphological Functions"""
    ###############################################

    # inverse Fourier transform
    def morphologicalActions(self, option):
        if len(self.grayImage) != 0:
            SE = np.array([[0,1,1,1,0],
                          [1,1,1,1,1],
                          [1,1,1,1,1],
                          [1,1,1,1,1],
                          [0,1,1,1,0]])
            image = binaryImage(self.grayImage)
            if option == 'erosion':
                result = erosion(image, SE)
            elif option == 'dilation':
                result = dilation(image, SE)
            elif option == 'opening':
                result = opening(image, SE)
            elif option == 'closing':
                result = closing(image, SE)

            self.drawImage(result)
