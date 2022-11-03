from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import math
import matplotlib.image as mpimg
import pydicom as dicom
from PIL import Image
class ImageViewer(FigureCanvasQTAgg):
    def __init__(self, parent=None, axisExisting=False, axisColor="#329da8"):
        self.fig = Figure(figsize = (6, 6), dpi = 80)
        self.axes = self.fig.add_subplot(111)

        # Variables
        self.loaded = False
        self.defaultImage = np.array([0])
        self.grayImage = np.array([0])
        self.imageTshape = np.array([0])
        self.axisExisting = axisExisting
        self.axisColor = axisColor

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
            imgData = dicom.dcmread(image_path, force=True)
            self.defaultImage = imgData.pixel_array
            self.axes.imshow(self.defaultImage)
        else:
            self.defaultImage = mpimg.imread(image_path)
            self.axes.imshow(self.defaultImage)
            imgData = Image.open(image_path)
       
        self.loaded = True
        self.draw()

        # If image is RGB transform it to gray.
        if self.defaultImage.ndim > 2:
            self.grayImage = self.defaultImage[:,:,0]
        else:
            self.grayImage = self.defaultImage

        # Depends on extension of the file
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
        
        self.axes.imshow(resizedImage[:zoomWidth,:zoomHeight], cmap="gray")
        self.draw()

        return resizedImage.shape[0], resizedImage.shape[1]

    # Construct T shape
    def constructT(self, background="white"):
        self.imageTshape = np.zeros((128,128), dtype=np.uint8)
        
        if background == "black":
            self.imageTshape.fill(255)

        for i in range(29,50):
            for j in range(29,100):
                self.imageTshape[i,j] = 1
        
        for i in range(49,100):
            for j in range(54,74):
                self.imageTshape[i,j] = 1

        self.axes.imshow(self.imageTshape, cmap="gray", extent=(0, 128, 0, 128))
        self.loaded = True
        self.draw()
    
    # Rotate T image
    def rotateT(self, angle, mode):
        if self.loaded:
            # Converting degrees to radians
            angle = math.radians(angle)

            cosine = math.cos(angle)
            sine = math.sin(angle)

            # Define the height of the image
            oldWidth = self.imageTshape.shape[0]
            # Define the width of the image
            oldHeight = self.imageTshape.shape[1]
          
            # Initilize rotated image 
            rotatedImage = np.zeros((oldWidth,oldHeight)) 

            # Find the center of the Rotated T image
            centerHeight = int( (oldHeight+1)/2) # mid row
            centerWidth = int( (oldWidth+1)/2 ) # mid col

            for i in range(oldWidth):
                for j in range(oldHeight):
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
                        if (x >= 0 and y >= 0 and x < oldWidth and y < oldHeight):
                            rotatedImage[j][i] = self.imageTshape[y][x]
        
                    elif mode == "linear":    
                        # Calculate the coordinate values for 4 surrounding pixels.
                        x_floor = math.floor(x)
                        x_ceil = min(oldWidth-1, math.ceil(x))
                        y_floor = math.floor(y)
                        y_ceil = min(oldHeight - 1, math.ceil(y))
                        
                        if (x >= 0 and y >= 0 and x < oldWidth and y < oldHeight):
                            if (x_ceil == x_floor) and (y_ceil == y_floor):
                                q = self.imageTshape[int(y), int(x)]
                            elif (y_ceil == y_floor):
                                q1 = self.imageTshape[int(y), int(x_floor)]
                                q2 = self.imageTshape[int(y), int(x_ceil)]
                                q = q1 * (x_ceil - x) + q2 * (x - x_floor)
                            elif (x_ceil == x_floor):
                                q1 = self.imageTshape[int(y_floor), int(x)]
                                q2 = self.imageTshape[int(y_ceil), int(x)]
                                q = (q1 * (y_ceil - y)) + (q2 * (y - y_floor))
                            else:
                                p1 = self.imageTshape[y_floor, x_floor]
                                p2 = self.imageTshape[y_ceil, x_floor]
                                p3 = self.imageTshape[y_floor, x_ceil]
                                p4 = self.imageTshape[y_ceil, x_ceil]

                                q1 = p1 * (y_ceil - y) + p2 * (y - y_floor)
                                q2 = p3 * (y_ceil - y) + p4 * (y - y_floor)
                                q = q1 * (x_ceil - x) + q2 * (x - x_floor)

                            rotatedImage[j][i] = q

            self.axes.imshow(rotatedImage, cmap="gray", extent=(0, 128, 0, 128))
            self.draw()

            return rotatedImage.shape[0],rotatedImage.shape[1]
        else:
            return "N/A","N/A"

    # shear T image
    def shearT(self, angle):
        if self.loaded:
            # Converting degrees to radians
            angle = math.radians(angle)

            # Define the height of the image
            oldWidth = self.imageTshape.shape[0]
            # Define the width of the image
            oldHeight = self.imageTshape.shape[1]
          
            # Initilize rotated image
            shearedImage = np.zeros((oldWidth,oldHeight))
            tangent = math.tan(-angle)

            # Find the center of the Rotated T image
            centerHeight = int( (oldHeight+1)/2) # mid row
            centerWidth = int( (oldWidth+1)/2 ) # mid col
            
            for i in range(oldWidth):
                for j in range(oldHeight):

                    x = (i-centerWidth)
                    y = (j-centerHeight)
                    
                    new_x = round(x-y*tangent)
                    new_y = y

                    # Add offset
                    new_x += centerHeight
                    new_y += centerWidth    
                    if (new_x >= 0 and new_y >= 0 and new_x < oldWidth and new_y < oldHeight):
                        shearedImage[j][i] = self.imageTshape[new_y,new_x]

            self.axes.imshow(shearedImage, cmap="gray", extent=(0, 128, 0, 128))
            self.draw()

    # Clear figure
    def clearImage(self):
        self.axes.clear()
        self.setShape()
        self.draw()

        self.loaded = False
        self.defaultImage = np.array([])
        self.grayImage = np.array([])