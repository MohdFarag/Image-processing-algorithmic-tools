from math import ceil, floor, sqrt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.image as mpimg
import pydicom as dicom
from PIL import Image


class ImageViewer(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 6),dpi=80)
        self.axes = self.fig.add_subplot(111)

        self.axes.grid(False)
        self.setTheme()

        # Variables
        self.loaded = False
        self.defaultImage = None
        self.grayImage = None

        super(ImageViewer, self).__init__(self.fig)   

    def setTheme(self):
        self.fig.set_edgecolor("black")
        
        self.axes.spines['bottom'].set_color('#329da8')
        self.axes.spines['top'].set_color('#329da8')
        self.axes.spines['right'].set_color('#329da8')
        self.axes.spines['left'].set_color('#329da8')

        self.axes.set_xticks([])
        self.axes.set_yticks([])

    def setImage(self, image_path, fileExtension):
        # Reading the image
        if fileExtension == "dcm":
            dicomImg = dicom.dcmread(image_path, force=True)
            self.defaultImage = dicomImg.pixel_array
            self.axes.imshow(self.defaultImage)
        else:
            self.defaultImage = mpimg.imread(image_path)
            self.axes.imshow(self.defaultImage)
            imgData = Image.open(image_path)

        # If image is RGB transform it to gray.
        if self.defaultImage.ndim > 2:
            self.grayImage = self.defaultImage[:,:,0]
        else:
            self.grayImage = self.defaultImage
        
        self.loaded = True
        self.draw()

        # Depends on extension of the file
        if fileExtension == "dcm":
            return dicomImg
        else:
            return imgData

    # Return to default scale
    def toDefaultScale(self):
        if self.loaded:
            # Load original Image
            self.axes.imshow(self.defaultImage)
            self.draw()

    # Transform to gray scale
    def toGrayScale(self):
        # Load gray image
        if self.loaded:
            self.axes.imshow(self.grayImage, cmap="gray")
            self.draw()

    # Nearest Neighbor Interpolation
    def nearestNeighborInterpolation(self, zoomingFactor):
        if self.loaded and self.grayImage.ndim == 2:      
                zoomingFactor = float(zoomingFactor)

                # Set size of zoomed image
                oldWidth = self.grayImage.shape[0]
                oldHeight = self.grayImage.shape[1]
                
                newWidth = ceil(oldWidth * zoomingFactor)
                newHeight = ceil(oldHeight * zoomingFactor)

                # Initilize the zoomed image
                zoomedImage = np.zeros((newWidth,newHeight))

                # Set the values
                for i in range(newWidth):
                    for j in range(newHeight):
                        # If I want to know the value of pixel at (3,1) then divide (3/2,1/2) 🡪 floor(1.5,0.5) 🡪 (1,0)
                        x = self.roundNum(i/zoomingFactor)
                        y = self.roundNum(j/zoomingFactor)

                        zoomedImage[i,j] = self.grayImage[x,y]
                
                zoomWidth = int(zoomedImage.shape[0]/zoomingFactor)
                zoomHeight = int(zoomedImage.shape[1]/zoomingFactor)
                
                self.axes.imshow(zoomedImage[:zoomWidth,:zoomHeight], cmap="gray")
                self.draw()

                return zoomedImage.shape

    # Linear Interpolation
    def linearInterpolation(self, zoomingFactor):
        if self.loaded and self.grayImage.ndim == 2:      
            zoomingFactor = float(zoomingFactor)

            # Get size of original image
            oldWidth = self.grayImage.shape[0]
            oldHeight = self.grayImage.shape[1]
            
            # Calc. size for zoomed image
            newWidth = ceil(oldWidth * zoomingFactor)
            newHeight = ceil(oldHeight * zoomingFactor)

            print(newWidth,newHeight)
            # Initilize the zoomed image
            zoomedImage = np.zeros((newWidth,newHeight))
            for i in range(newWidth):
                for j in range(newHeight):
                    # Relative coordinates of the pixel in output space
                    x_out = j / newWidth
                    y_out = i / newHeight

                    # Corresponding absolute coordinates of the pixel in input space
                    x_in = (x_out * oldWidth)
                    y_in = (y_out * oldHeight)

                    # Nearest neighbours coordinates in input space
                    x_prev = int(np.floor(x_in))
                    x_next = x_prev + 1
                    y_prev = int(np.floor(y_in))
                    y_next = y_prev + 1

                    # Sanitize bounds - no need to check for < 0
                    x_prev = min(x_prev, oldWidth - 1)
                    x_next = min(x_next, oldWidth - 1)
                    y_prev = min(y_prev, oldHeight - 1)
                    y_next = min(y_next, oldHeight - 1)
                    
                    # Distances between neighbour nodes in input space
                    Dy_next = y_next - y_in
                    Dy_prev = 1. - Dy_next; # because next - prev = 1
                    Dx_next = x_next - x_in
                    Dx_prev = 1. - Dx_next; # because next - prev = 1
                    
                    try:
                        zoomedImage[i][j] = Dy_prev * (self.grayImage[y_next][x_prev] * Dx_next + self.grayImage[y_next][x_next] * Dx_prev) \
                        + Dy_next * (self.grayImage[y_prev][x_prev] * Dx_next + self.grayImage[y_prev][x_next] * Dx_prev)
                    except Exception as e:
                        print(e)
                        
            zoomWidth = int(zoomedImage.shape[0]/zoomingFactor)
            zoomHeight = int(zoomedImage.shape[1]/zoomingFactor)            
            
            self.axes.imshow(zoomedImage[:zoomWidth,:zoomHeight], cmap="gray")
            
            self.draw()

            return zoomedImage.shape
    
    # Round in Nearest Neighbor Interpolation
    def roundNum(self, num):
        if num - int(num) <= 0.5:
            return floor(num)
        else:
            return floor(num)

    # Clear figure
    def clearImage(self):
        self.axes.clear()
        self.setTheme()
        self.draw()

        self.loaded = False
        self.defaultImage = None
        self.grayImage = None