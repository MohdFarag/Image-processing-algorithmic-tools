from math import ceil, floor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.image as mpimg
import pydicom as dicom
from PIL import Image

class ImageViewer(FigureCanvasQTAgg):
    def __init__(self, parent=None, axisExisting=False, axisColor="#329da8"):
        self.fig = Figure(figsize = (6, 6), dpi = 80)
        self.axes = self.fig.add_subplot(111)

        # Variables
        self.loaded = False
        self.defaultImage = None
        self.grayImage = None
        self.axisExisting = axisExisting
        self.axisColor = axisColor

        self.axes.grid(self.axisColor)
        self.axes.grid(False)
        self.setShape()

        super(ImageViewer, self).__init__(self.fig)
   
    # Set Theme
    def setShape(self):
        self.fig.set_edgecolor("black")
        self.axes.spines['bottom'].set_color(self.axisColor)
        self.axes.spines['top'].set_color(self.axisColor)
        self.axes.spines['right'].set_color(self.axisColor)
        self.axes.spines['left'].set_color(self.axisColor)

        if not self.axisExisting:
            self.axes.set_xticks([])
            self.axes.set_yticks([])

    # Set Image
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
                
                newWidth = round(oldWidth * zoomingFactor)
                newHeight = round(oldHeight * zoomingFactor)

                # Initilize the zoomed image
                zoomedImage = np.zeros((newWidth,newHeight))

                # Set the values
                for i in range(newWidth):
                    for j in range(newHeight):
                        if i/zoomingFactor > oldWidth - 1 or j/zoomingFactor > oldHeight - 1 :
                            # If I want to know the value of pixel at (3,1) then divide (3/2,1/2) 🡪 floor(1.5,0.5) 🡪 (1,0)
                            x = floor(i/zoomingFactor)
                            y = floor(j/zoomingFactor)
                        else:
                            # If I want to know the value of pixel at (3,1) then divide (3/2,1/2) 🡪 floor(1.5,0.5) 🡪 (1,0)
                            x = round(i/zoomingFactor)
                            y = round(j/zoomingFactor)

                        zoomedImage[i,j] = self.grayImage[x,y]


                zoomWidth = int(zoomedImage.shape[0]/zoomingFactor)
                zoomHeight = int(zoomedImage.shape[1]/zoomingFactor)
                
                self.axes.imshow(zoomedImage[:zoomWidth,:zoomHeight], cmap="gray")
                self.draw()

                return zoomedImage.shape[1],zoomedImage.shape[0]
        else:
            return "N/A","N/A"

    # Linear Interpolation Slow
    def linearInterpolation(self, scale_factor):
        if self.loaded and self.grayImage.ndim == 2:
            scale_factor = float(scale_factor)

            img_height, img_width = self.grayImage.shape[:2]
            height = round(img_height*scale_factor)
            width = round(img_width*scale_factor)

            resized = np.zeros([height, width])

            x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
            y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

            for i in range(height):
                for j in range(width):
                    x_l, y_l = floor(x_ratio * j), floor(y_ratio * i)
                    x_h, y_h = ceil(x_ratio * j), ceil(y_ratio * i)

                    x_weight = (x_ratio * j) - x_l
                    y_weight = (y_ratio * i) - y_l
                    
                    try:
                        a = self.grayImage[y_l, x_l]
                        b = self.grayImage[y_l, x_h]
                        c = self.grayImage[y_h, x_l]
                        d = self.grayImage[y_h, x_h]
                            
                        pixel = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight
                        resized[i][j] = pixel
                    except:
                        pass

            zoomWidth = int(resized.shape[0]/scale_factor)
            zoomHeight = int(resized.shape[1]/scale_factor)
            self.axes.imshow(resized, cmap="gray")
            self.draw()

            return resized.shape
        else:
            return "N/A","N/A"
    
    # Linear Interpolation Vectorized
    def linearInterpolationVectorized(self, scale_factor):
        if self.loaded and self.grayImage.ndim == 2:      
            scale_factor = round(scale_factor,2)

            img_height, img_width = self.grayImage.shape[:2]
            image = self.grayImage.ravel()

            height = round(img_height*scale_factor)
            width = round(img_width*scale_factor)

            x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
            y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

            y, x = np.divmod(np.arange(height * width), width)

            x_l = np.floor(x_ratio * x).astype('int32')
            y_l = np.floor(y_ratio * y).astype('int32')

            x_h = np.ceil(x_ratio * x).astype('int32')
            y_h = np.ceil(y_ratio * y).astype('int32')

            x_weight = (x_ratio * x) - x_l
            y_weight = (y_ratio * y) - y_l

            a = image[y_l * img_width + x_l]
            b = image[y_l * img_width + x_h]
            c = image[y_h * img_width + x_l]
            d = image[y_h * img_width + x_h]

            resized = a * (1 - x_weight) * (1 - y_weight) + \
                        b * x_weight * (1 - y_weight) + \
                        c * y_weight * (1 - x_weight) + \
                        d * x_weight * y_weight

            resized = resized.reshape(height, width)

            zoomWidth = int(resized.shape[0]/scale_factor)
            zoomHeight = int(resized.shape[1]/scale_factor)

            self.axes.imshow(resized[:zoomWidth,:zoomHeight], cmap="gray")
            self.draw()

            return resized.shape[1],resized.shape[0]

    def constructT(self):
        imageTshape = np.zeros((128,128))
        for i in range(29,50):
            for j in range(29,100):
                imageTshape[i,j] = 1
        
        for i in range(49,100):
            for j in range(54,74):
                imageTshape[i,j] = 1

        self.axes.imshow(imageTshape, cmap="gray", extent=(0, 128, 0, 128))
        self.draw()
        

    # Clear figure
    def clearImage(self):
        self.axes.clear()
        self.setShape()
        self.draw()

        self.loaded = False
        self.defaultImage = None
        self.grayImage = None