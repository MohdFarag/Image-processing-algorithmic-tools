# matplotlib
from matplotlib import pyplot as plt
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
        self.img = np.array([])
        self.axes.grid(False)
        self.axes.set_axis_off()

        super(ImageViewer, self).__init__(self.fig)   


    def setImage(self, image_path, fileExtension):
        self.clearImage()
        if image_path == "":
            return

        if fileExtension == "dcm":
            dicomImg = dicom.dcmread(image_path)
            self.img = dicomImg.pixel_array
            self.axes.imshow(self.img,cmap="gray")
        else:
            self.img = mpimg.imread(image_path)
            self.axes.imshow(self.img)
            imgData = Image.open(image_path)

        self.draw()
        self.axes.set_axis_off()

        if fileExtension == "dcm":
            return dicomImg
        else:
            return imgData

    def clearImage(self):
        self.axes.clear()
        self.axes.set_axis_off()