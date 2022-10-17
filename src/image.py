# matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.image as mpimg
import pydicom as dicom
from PIL import Image


class ImageViewer(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111,)
        self.axes.grid(False)
        self.setTheme()

        # Variables
        self.loaded = False
        self.img = None
        self.newImage = None

        super(ImageViewer, self).__init__(self.fig)   

    def setTheme(self):
        self.fig.set_edgecolor("black")
        
        self.axes.spines['bottom'].set_color('red')
        self.axes.spines['top'].set_color('red')
        self.axes.spines['right'].set_color('red')
        self.axes.spines['left'].set_color('red')
        self.axes.set_xticks([])
        self.axes.set_yticks([])


    def setImage(self, image_path, fileExtension):
        if fileExtension == "dcm":
            dicomImg = dicom.dcmread(image_path, force=True)
            self.img = dicomImg.pixel_array
            self.axes.imshow(self.img)
        else:
            self.img = mpimg.imread(image_path)
            self.axes.imshow(self.img)
            imgData = Image.open(image_path)

        self.newImage = np.dot(self.img[...,:3], [0.299, 0.587, 0.144])
        self.draw()
        self.loaded = True

        if fileExtension == "dcm":
            return dicomImg
        else:
            return imgData

    # Return to default scale
    def toDefaultScale(self):
        if self.loaded:
            self.axes.imshow(self.img)
            self.draw()

    # Transform to gray scale
    def toGrayScale(self):
        if self.loaded:
            try:
                self.axes.imshow(self.newImage, cmap='gray')
            except:
                self.axes.imshow(self.img, cmap='gray')
            finally:    
                self.draw()

    def nearestNeighborInterpolation(self, zoomingFactor):
        print(zoomingFactor)
        if self.loaded:
            self.axes.imshow(self.newImage)
            self.draw()

    def linearInterpolation(self, zoomingFactor):
        print(zoomingFactor)
        if self.loaded:
            self.axes.imshow(self.newImage)
            self.draw()

    def clearImage(self):
        self.axes.clear()
        self.setTheme()
        self.draw()

        self.loaded = False
        self.img = None
        self.newImage = None
