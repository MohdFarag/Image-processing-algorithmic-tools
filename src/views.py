from math import ceil, log2
import cv2 as cv
import numpy as np

# Resources
from .rcIcon import *

# Importing sys package
import sys
import os

# Import Classes
from .additionsQt import *
from .image import ImageViewer

# Importing Qt widgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# Importing Logging
from .log import appLogger

# Mode/Color to Bit dipth dict
MODE_TO_BPP = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}

# Window class
class Window(QMainWindow):
    """Main Window"""
    def __init__(self):
        """Initializer."""
        super().__init__()

        ### Variables
        self.fileExtension = ""
        self.widthOfImage = ""
        self.heightOfImage = ""
        self.sizeOfImage = ""
        self.depthOfImage = ""
        self.modeOfImage = ""
        self.modalityOfImage = ""
        self.nameOfPatient = ""
        self.ageOfPatient = ""
        self.bodyOfPatient = ""    
        
        ### Setting Icon
        self.setWindowIcon(QIcon(":icon.svg"))

        ### Setting title
        self.setWindowTitle("Image Viewer")

        ### UI contents
        self._createActions()
        self._createMenuBar()
        self._createToolBar()
        self._createToolBar("zoom")
        self._createStatusBar()
        # Central area
        self._initUI()
        # Connect signals
        self._connect()
   
    # Actions
    def _createActions(self):
        # Open Action
        self.openAction = QAction(QIcon(":file.ico"), "&Open Image...", self)
        self.openAction.setShortcut("Ctrl+O")
        self.openAction.setStatusTip('Open a new image')

        # Grayscale Action
        self.clearAction = QAction(QIcon(":clear.png"), "&Close Image", self)
        self.clearAction.setShortcut("Ctrl+C")
        self.clearAction.setStatusTip('Close the image')

        # Default scale Action
        self.defaultScaleAction = QAction(QIcon(":defscale.png"), "&Default Scale", self)
        self.defaultScaleAction.setShortcut("Ctrl+D")
        self.defaultScaleAction.setStatusTip('Return to default scale')

        # Grayscale Action
        self.grayScaleAction = QAction(QIcon(":grscale.png"), "&Gray Scale", self)
        self.grayScaleAction.setShortcut("Ctrl+G")
        self.grayScaleAction.setStatusTip('Transform to gray scale')

        # Zoom Nearest Neighbor Interpolation Action
        self.zoomNearestNeighborInterpolationAction = QAction(QIcon(":paxiliteZoom.png"), "&Nearest Neighbor", self)
        self.zoomNearestNeighborInterpolationAction.setShortcut("Ctrl+1")
        self.zoomNearestNeighborInterpolationAction.setStatusTip('Zoom in/out by Nearest Neighbor Interpolation method based on input')

        # Zoom Linear Interpolation Action
        self.zoomLinearInterpolationAction = QAction(QIcon(":zoom.png"), "&Linear", self)
        self.zoomLinearInterpolationAction.setShortcut("Ctrl+2")
        self.zoomLinearInterpolationAction.setStatusTip('Zoom in/out by Linear Interpolation method based on input')

        # Exit Action
        self.exitAction = QAction(QIcon(":exit.svg"), "&Exit", self)
        self.exitAction.setShortcut("Ctrl+Q")
        self.exitAction.setStatusTip('Exit application')

        # Help Actions
        self.helpContentAction = QAction("&Help Content", self)
        self.helpContentAction.setStatusTip('Help')
        self.checkUpdatesAction = QAction("&Check For Updates", self)
        self.checkUpdatesAction.setStatusTip('Check Updates')
        self.aboutAction = QAction("&About...", self)
        self.aboutAction.setStatusTip('About')

    # Add seperator
    def addSeperator(self, parent):
        # Creating a separator action
        self.separator = QAction(self)
        self.separator.setSeparator(True)
        parent.addAction(self.separator)

    # Menu
    def _createMenuBar(self):
        # Menu bar
        menuBar = self.menuBar()

        ## File tap
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.openAction) # Open file in menu
        fileMenu.addSeparator() # Seperator
        fileMenu.addAction(self.clearAction) # Clear image in menu
        fileMenu.addSeparator() # Seperator
        fileMenu.addAction(self.exitAction) # Exit file in menu

        ## Edit tap
        editMenu = QMenu("&Edit", self)
        editMenu.addAction(self.defaultScaleAction) # Default scale in menu
        editMenu.addAction(self.grayScaleAction) # Gray scale in menu
        
        ## Help tap
        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addSeparator() # Seperator
        helpMenu.addAction(self.checkUpdatesAction)
        helpMenu.addSeparator() # Seperator
        helpMenu.addAction(self.aboutAction)
        
        ## Append taps
        menuBar.addMenu(fileMenu)
        menuBar.addMenu(editMenu)
        menuBar.addMenu(helpMenu)

    # Tool Bar
    def _createToolBar(self, type=""):
        self.toolBar = QToolBar("Tool Bar")
        if type=="zoom":
            self.addToolBar(Qt.TopToolBarArea,self.toolBar)
            self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            self.zoomFactorInput = QLineEdit("0")
            self.toolBar.addWidget(self.zoomFactorInput)
            self.toolBar.addAction(self.zoomNearestNeighborInterpolationAction)
            self.toolBar.addAction(self.zoomLinearInterpolationAction)
        else:
            # Using a title
            self.addToolBar(Qt.RightToolBarArea, self.toolBar)
            self.toolBar.addAction(self.openAction)
            self.toolBar.addAction(self.defaultScaleAction)
            self.toolBar.addAction(self.grayScaleAction)
            self.toolBar.addAction(self.clearAction)
    
    # Context Menu Event
    def contextMenuEvent(self, event):
        # Creating a menu object with the central widget as parent
        menu = QMenu(self)
        # Populating the menu with actions
        menu.addAction(self.openAction)
        self.addSeperator(menu)
        menu.addAction(self.defaultScaleAction)
        menu.addAction(self.grayScaleAction)
        menu.addAction(self.clearAction)
        self.addSeperator(menu)
        menu.addAction(self.helpContentAction)
        menu.addAction(self.checkUpdatesAction)
        menu.addAction(self.aboutAction)
        # Launching the menu
        menu.exec(event.globalPos())
    
    # Status Bar
    def _createStatusBar(self):
        self.statusbar = self.statusBar()
        self.statusbar.setStyleSheet(f"""font-size:15px;
                                 padding: 4px;""")
        self.statusbar.showMessage("Ready", 3000)
        # Adding a permanent message
        self.statusbar.addPermanentWidget(QLabel("Upload your image"))

    # GUI
    def _initUI(self):
        centralMainWindow = QWidget(self)
        self.setCentralWidget(centralMainWindow)
        # Outer Layout
        outerLayout = QVBoxLayout()

        # Initialize tab screen
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""font-size:15px;""")
        ### init GUI ###
        
        # Main layout
        self.originalTab = QWidget()
        self.originalLayout()
        tabs.addTab(self.originalTab, "Original Image")
        
        # Main layout
        self.zoomTab = QWidget()
        self.zoomLayout()
        tabs.addTab(self.zoomTab, "Zoom Image")
    
        outerLayout.addWidget(tabs)

        # Add docker
        self.addDockLayout()
    
        ### GUI ###
        centralMainWindow.setLayout(outerLayout)

    # Original Layout
    def originalLayout(self):
        originalLayout = QVBoxLayout()
        
        self.originalViewer = ImageViewer()
        originalLayout.addWidget(self.originalViewer)

        self.originalTab.setLayout(originalLayout)

    # Zoom Layout
    def zoomLayout(self):
        zoomLayout = QVBoxLayout()
        
        self.zoomViewer = ImageViewer()
        zoomLayout.addWidget(self.zoomViewer)

        self.zoomTab.setLayout(zoomLayout)

    # Dock widget 
    def addDockLayout(self):   
        self.dockInfo = QDockWidget("Information", self)
        # Tree widget which contains the info
        self.dataWidget = QTreeWidget()
        self.dataWidget.setColumnCount(2) # Att, Val
        self.dataWidget.setHeaderLabels(["Attribute", "Value"])
        self.setInfo("dcm") # Default

        self.dockInfo.setWidget(self.dataWidget)
        self.dockInfo.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockInfo)

    # Set information
    def setInfo(self, ext, width="", height="", size="", depth="", color="", Modality="", PatientName="", PatientAge="", BodyPartExamined=""):
        info = dict() # Initilize the dicom
        if ext == "dcm":
            info = {
                    "Width":width, 
                    "Height":height,
                    "Total Size":size,
                    "Bit depth":depth,
                    "Image color":color,
                    "Modality used": Modality,
                    "Patient name": PatientName,
                    "Patient Age": PatientAge,
                    "Body part examined": BodyPartExamined
                }
        elif ext in ["Bilinear", "Nearest Neighbor"]:
            info = {
                    "Interpolation Type": ext,
                    "Width": width, 
                    "Height": height,
                    "Image color": f"{color}->Grayscale" ,
                }
        else:
            info = {
                    "Width":width, 
                    "Height":height,
                    "Total Size":size,
                    "Bit depth":depth,
                    "Image color":color
                }
        
        # Update the tree
        self.setDataOfTree(info)

    # Set the data of the tree
    def setDataOfTree(self, data):
        self.dataWidget.clear()
        item = QTreeWidgetItem(["metadata"])
        for key, value in data.items():
            child = QTreeWidgetItem([key, str(value)])
            item.addChild(child)
        self.dataWidget.insertTopLevelItem(0, item)
    
    # Connect
    def _connectActions(self):
        self.openAction.triggered.connect(self.browseImage) # When click on browse image action
        self.defaultScaleAction.triggered.connect(self.originalViewer.toDefaultScale) # When click on grayscale image action
        self.grayScaleAction.triggered.connect(self.originalViewer.toGrayScale) # When click on grayscale image action
        self.clearAction.triggered.connect(self.originalViewer.clearImage) # When click on exit action
        self.clearAction.triggered.connect(self.zoomViewer.clearImage) # When click on exit action
        
        # Interpoloations
        self.zoomNearestNeighborInterpolationAction.triggered.connect(self.nearestNeighborInterpolation)
        self.zoomLinearInterpolationAction.triggered.connect(self.linearInterpolation)

        self.exitAction.triggered.connect(self.exit) # When click on exit action
    
    def _connect(self):
        self._connectActions()

    # Linear Interpolation
    def linearInterpolation(self):
        try:
            zoomingFactor = float(self.zoomFactorInput.text())
        except:
            QMessageBox.critical(self , "Invalid zooming factor" , "Please enter valid facor.")
            return
        
        if zoomingFactor > 0:
            self.widthOfImage, self.heightOfImage = self.zoomViewer.linearInterpolation(zoomingFactor)
            self.widthOfImage = f"{self.widthOfImage} px"
            self.heightOfImage = f"{self.heightOfImage} px"
            self.fileExtension = "Bilinear"
            self.setInfo(self.fileExtension, self.widthOfImage, self.heightOfImage, self.sizeOfImage, self.depthOfImage, self.modeOfImage, self.modalityOfImage, self.nameOfPatient, self.ageOfPatient, self.bodyOfPatient)
        else:
            QMessageBox.critical(self , "Invalid zooming factor" , "Please enter valid facor.")

    # Nearest Neighbor Interpolation
    def nearestNeighborInterpolation(self):
        try:
            zoomingFactor = float(self.zoomFactorInput.text())
        except:
            QMessageBox.critical(self , "Invalid Zooming Factor" , "Please Enter Valid Factor.")
            return

        if zoomingFactor > 0:
            self.widthOfImage, self.heightOfImage = self.zoomViewer.nearestNeighborInterpolation(zoomingFactor)
            self.widthOfImage = f"{self.widthOfImage} px"
            self.heightOfImage = f"{self.heightOfImage} px"
            self.fileExtension = "Nearest Neighbor"
            self.setInfo(self.fileExtension, self.widthOfImage, self.heightOfImage, self.sizeOfImage, self.depthOfImage, self.modeOfImage, self.modalityOfImage, self.nameOfPatient, self.ageOfPatient, self.bodyOfPatient)
        else:
            QMessageBox.critical(self , "Invalid Zooming Factor" , "Please Enter Valid Factor.")

    # Open image
    def browseImage(self):
        # Browse Function
        path, _ = QFileDialog.getOpenFileName(None, "Load Image File", filter="Custom files (*.bmp *.jpeg *.jpg *.dcm);;All files (*.*)")            
        self.fileExtension = path.split(".")[-1] # get ext.
        
        # If no image choosed
        if path == "":
            return

        try:
            data = self.originalViewer.setImage(path, self.fileExtension)
            data = self.zoomViewer.setImage(path, self.fileExtension)
            self.zoomViewer.toGrayScale()
        except:
            # Error
            appLogger.exception("Can't open the file !")
            QMessageBox.critical(self , "Corrupted image" , "Can't open the file !")
        else:
            self.statusbar.showMessage(path.split("/")[-1])
            
            if self.fileExtension == "dcm":
                # If dicom
                self.widthOfImage =  self.getAttr(data, 'Columns')
                self.heightOfImage = self.getAttr(data, 'Rows')
                self.depthOfImage = self.getAttr(data, 'BitsAllocated')
                self.sizeOfImage = f"{self.widthOfImage * self.heightOfImage * self.depthOfImage} bits" 
                self.widthOfImage =  f"{self.widthOfImage} px"
                self.heightOfImage = f"{self.heightOfImage} px"
                self.depthOfImage = f"{self.depthOfImage} bit/pixel"
                self.modeOfImage = self.getAttr(data, "PhotometricInterpretation")
                self.modalityOfImage = self.getAttr(data, "Modality")
                self.nameOfPatient = self.getAttr(data, "PatientName")
                self.ageOfPatient = self.getAttr(data,"PatientAge")
                self.bodyOfPatient = self.getAttr(data,"BodyPartExamined") 

                # Set the information                 
                self.setInfo(self.fileExtension, self.widthOfImage, self.heightOfImage, self.sizeOfImage, self.depthOfImage, self.modeOfImage, self.modalityOfImage, self.nameOfPatient, self.ageOfPatient, self.bodyOfPatient)
            else:
                # If (jpeg, bitmap)
                imageChannel = cv.imread(path)

                self.widthOfImage = self.getAttr(data,'width')
                self.heightOfImage = self.getAttr(data,'height')
                self.depthOfImage = self.getDepth(data, imageChannel)

                self.sizeOfImage = f"{self.widthOfImage * self.heightOfImage * self.depthOfImage} bits" 
                self.widthOfImage =  f"{self.widthOfImage} px"
                self.heightOfImage = f"{self.heightOfImage} px"
                self.depthOfImage = f"{self.depthOfImage} bit/pixel"
                self.modeOfImage = self.getAttr(data,"mode")
                
                # Set the information
                self.setInfo(self.fileExtension, self.widthOfImage, self.heightOfImage, self.sizeOfImage, self.depthOfImage, self.modeOfImage, self.modalityOfImage, self.nameOfPatient, self.ageOfPatient, self.bodyOfPatient)

    # Get Data
    def getAttr(self, variable, att):
        if hasattr(variable, att):
            # If attribute is found.
            return getattr(variable, att)
        else:
            # If attribute is not found.
            return "N/A"

    # TODO: Check: Right or Wrong
    def getDepth(self, image, imageChannel):
        image_sequence = image.getdata()
        image_array = np.asarray(image_sequence) 
        range = image_array.max() - image_array.min()
        bitDepthForOneChannel = ceil(log2(range))
        
        try:
            _ , _ , numOfChannels = imageChannel.shape
        except:
            numOfChannels = 1
        finally:
            bitDepth = bitDepthForOneChannel * numOfChannels
            return bitDepth

    # Exit the application
    def exit(self):
        exitDialog = QMessageBox.critical(self,
        "Exit the application",
        "Are you sure you want to exit the application?",
        buttons=QMessageBox.Yes | QMessageBox.No,
        defaultButton=QMessageBox.No)

        if exitDialog == QMessageBox.Yes:
            # Exit the application
            sys.exit()