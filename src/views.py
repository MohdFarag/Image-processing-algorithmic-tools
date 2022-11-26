from math import ceil, log2
import cv2 as cv
import numpy as np

# Resources
from .rcIcon import *

# Importing sys package
import sys

# Import Classes
from .additionsQt import *
from .image import ImageViewer

# Importing Qt widgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# Importing Logging
from .log import appLogger

# Window class
class Window(QMainWindow):
    """Main Window"""
    def __init__(self, *args, **kwargs):
        """Initializer."""
        super(Window, self).__init__(*args, **kwargs)

        ### Variables
        self.widthOfImage = 0
        self.heightOfImage = 0
        self.sizeOfImage = 0
        self.depthOfImage = 0
        self.fileExtension = "N/A"
        self.modeOfImage = "N/A"
        self.modalityOfImage = "N/A"
        self.nameOfPatient = "N/A"
        self.ageOfPatient = "N/A"
        self.bodyOfPatient = "N/A"
        self.interpolationMode = "N/A"
        
        ### Setting Icon
        self.setWindowIcon(QIcon(":icon.svg"))

        ### Setting title
        self.setWindowTitle("Image Processing Algorithms")

        ### UI contents
        self._createActions()
        self._createMenuBar()
        self._createToolBar("original")
        self._createToolBar("zoom")
        self._createToolBar("T")
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

        # Add tab Action
        self.addTabAction = QAction(QIcon(":add.ico"), "&Add new tab...", self)
        self.addTabAction.setShortcut("Ctrl+A")
        self.addTabAction.setStatusTip('Add a new tab')

        # Clear Action
        self.clearAction = QAction(QIcon(":clear.png"), "&Close Image", self)
        self.clearAction.setShortcut("Ctrl+C")
        self.clearAction.setStatusTip('Close the image')

        # Zoom Nearest Neighbor Interpolation Action
        self.zoomNearestNeighborInterpolationAction = QAction(QIcon(":paxiliteZoom.png"), "&Zoom Nearest Neighbor", self)
        self.zoomNearestNeighborInterpolationAction.setShortcut("Ctrl+1")
        self.zoomNearestNeighborInterpolationAction.setStatusTip('Zoom in/out by Nearest Neighbor Interpolation method based on input')

        # Zoom Linear Interpolation Action
        self.zoomLinearInterpolationAction = QAction(QIcon(":zoom.png"), "&Zoom Linear", self)
        self.zoomLinearInterpolationAction.setShortcut("Ctrl+2")
        self.zoomLinearInterpolationAction.setStatusTip('Zoom in/out by Linear Interpolation method based on input')

        # Construct T image Action
        self.constructTAction = QAction(QIcon(":T.png"), "&Construct T", self)
        self.constructTAction.setShortcut("ctrl+T")
        self.constructTAction.setStatusTip('Construct an image with a T letter in the center')

        # Rotate the image
        self.rotateNearestAction = QAction(QIcon(":rotate.png"), "&Rotate T Nearest", self)
        self.rotateNearestAction.setShortcut("ctrl+3")
        self.rotateNearestAction.setStatusTip('Rotate the image')

        # Rotate the image
        self.rotateLinearAction = QAction(QIcon(":rotate.png"), "&Rotate T Linear", self)
        self.rotateLinearAction.setShortcut("ctrl+4")
        self.rotateLinearAction.setStatusTip('Rotate the image')

        # Shear the image
        self.shearAction = QAction(QIcon(":shear.png"), "&Shear T", self)
        self.shearAction.setShortcut("ctrl+5")
        self.shearAction.setStatusTip('Shear the image')
        
        # Show histogram of the image
        self.showHistogramAction = QAction(QIcon(":histogram.png"), "&Histogram", self)
        self.showHistogramAction.setShortcut("ctrl+H")
        self.showHistogramAction.setStatusTip('Show the histogram')

        # Show histogram of the image
        self.save = QAction(QIcon(":save.png"), "&Save image", self)
        self.save.setShortcut("ctrl+s")
        self.save.setStatusTip('Save the image')
        
        # Equalize the image
        self.equalizeAction = QAction(QIcon(":equalize.png"), "&Equalize", self)
        self.equalizeAction.setShortcut("ctrl+E")
        self.equalizeAction.setStatusTip('Equalize the image')

        # Box filter
        self.unsharpAction = QAction(QIcon(":box.png"), "&Un-sharp Mask", self)
        self.unsharpAction.setShortcut("ctrl+U")
        self.unsharpAction.setStatusTip('Create a Unsharp Masking')

        self.addSaltNoiseAction = QAction(QIcon(":salt.png"), "&Add salt and pepper", self)
        self.addSaltNoiseAction.setShortcut("ctrl+P")
        self.addSaltNoiseAction.setStatusTip('Add salt and pepper noise')

        self.medianFilterAction = QAction(QIcon(":median.png"), "&Median filter", self)
        self.medianFilterAction.setShortcut("ctrl+M")
        self.medianFilterAction.setStatusTip('Delete salt and pepper noise')

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
        fileMenu.addAction(self.save)
        fileMenu.addSeparator() # Seperator
        fileMenu.addAction(self.clearAction) # Clear image in menu
        fileMenu.addSeparator() # Seperator
        fileMenu.addAction(self.exitAction) # Exit file in menu

        ## Edit tap
        editMenu = QMenu("&Edit", self)
        editMenu.addAction(self.constructTAction) # Construct T in menu
        editMenu.addAction(self.showHistogramAction) # Show histogram in menu
        editMenu.addAction(self.equalizeAction) # Equalize image in menu

        ## Filter
        filterMenu = QMenu("&Filter", self)
        filterMenu.addAction(self.unsharpAction)

        ## View tap
        viewMenu = QMenu("&View", self)
        viewMenu.addAction(self.addTabAction) # Add tab in menu
        
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
        menuBar.addMenu(filterMenu)
        menuBar.addMenu(viewMenu)
        menuBar.addMenu(helpMenu)

    # Tool Bar
    def _createToolBar(self, type=""):
        self.toolBar = QToolBar("Tool Bar")
        if type == "zoom":
            self.addToolBar(Qt.TopToolBarArea,self.toolBar)
            self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

            self.sizeInput = QLineEdit("1")
            self.sizeInput.setStyleSheet("""border:1px solid #00d; 
                                                height:18px; 
                                                padding:2px; 
                                                border-radius:5px; 
                                                font-size:16px; 
                                                margin-right:5px""")
            self.toolBar.addWidget(self.sizeInput)
            
            self.factorInput = QLineEdit("0")
            self.factorInput.setStyleSheet("""border:1px solid #00d; 
                                                height:18px; 
                                                padding:2px; 
                                                border-radius:5px; 
                                                font-size:16px; 
                                                margin-right:5px""")
            self.toolBar.addWidget(self.factorInput)

            self.toolBar.addAction(self.unsharpAction) 

            self.toolBar.addAction(self.zoomNearestNeighborInterpolationAction)
            self.toolBar.addAction(self.zoomLinearInterpolationAction)
            self.toolBar.addAction(self.rotateNearestAction)
            self.toolBar.addAction(self.rotateLinearAction)
            self.toolBar.addAction(self.shearAction)        
        
        elif type == "original":
            # Using a title
            self.addToolBar(Qt.RightToolBarArea, self.toolBar)  # type: ignore
            self.toolBar.addAction(self.openAction)
            self.toolBar.addAction(self.showHistogramAction)
            self.toolBar.addAction(self.equalizeAction)
            self.toolBar.addAction(self.addSaltNoiseAction)
            self.toolBar.addAction(self.medianFilterAction)
            self.toolBar.addAction(self.clearAction)
        elif type == "T":
            self.addToolBar(Qt.RightToolBarArea, self.toolBar)  # type: ignore
            self.toolBar.addAction(self.constructTAction)

    # Context Menu Event
    def contextMenuEvent(self, event):
        # Creating a menu object with the central widget as parent
        menu = QMenu(self)
        # Populating the menu with actions
        menu.addAction(self.openAction)
        menu.addAction(self.save)
        self.addSeperator(menu)
        menu.addAction(self.equalizeAction)
        menu.addAction(self.clearAction)
        self.addSeperator(menu)
        menu.addAction(self.constructTAction)
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
        self.statusbar.addPermanentWidget(QLabel("Image processing algorithms"))

    # GUI
    def _initUI(self):
        centralMainWindow = QWidget(self)
        self.setCentralWidget(centralMainWindow)
        # Outer Layout
        outerLayout = QVBoxLayout()

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.setStyleSheet(f"""font-size:16px;""")
        
        ### init GUI ###
        self.currentTab = self.addNewTab("Image", "red")
        outerLayout.addWidget(self.tabs)
        
        # Add docker
        self.addDockLayout()
        ### GUI ###

        centralMainWindow.setLayout(outerLayout)
   
    # Dock widget 
    def addDockLayout(self):   
        self.dockInfo = QDockWidget("Information", self)
        # Tree widget which contains the info
        self.dataWidget = QTreeWidget()
        self.dataWidget.setColumnCount(2) # Att, Val
        self.dataWidget.setHeaderLabels(["Attribute", "Value"])
        self.setInfo("jpeg") # Default

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
        elif ext in ["Zoom Bilinear", "Zoom Nearest Neighbor"]:
            info = {
                    "Interpolation Type": ext,
                    "Width": width, 
                    "Height": height,
                    "Image color": f"{color}->Grayscale" ,
                }
        elif ext in ["Rotate Bilinear", "Rotate Nearest Neighbor"]:
            info = {
                    "Interpolation Type": ext,
                    "Width": width, 
                    "Height": height,
                    "Angle": f"{size}°",
                    "Direction": depth,
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
        item = QTreeWidgetItem(["Data"])
        for key, value in data.items():
            child = QTreeWidgetItem([key, str(value)])
            item.addChild(child)
        self.dataWidget.insertTopLevelItem(0, item)
    
    # Connect
    def _connectActions(self):
        # Original Actions
        self.openAction.triggered.connect(self.browseImage) # When click on browse image action
        
        self.clearAction.triggered.connect(lambda: self.currentTab.primaryViewer.reset()) # When click on clear action
        self.clearAction.triggered.connect(lambda: self.currentTab.histogramViewer.reset()) # When click on clear action

        # Zoom image
        self.zoomNearestNeighborInterpolationAction.triggered.connect(lambda: self.zoomImage("nearest"))
        self.zoomLinearInterpolationAction.triggered.connect(lambda: self.zoomImage("linear"))

        # Construct T
        self.constructTAction.triggered.connect(lambda: self.currentTab.drawT())

        # Rotate image
        self.rotateNearestAction.triggered.connect(lambda: self.rotateImage(interpolationMode="nearest"))
        self.rotateLinearAction.triggered.connect(lambda: self.rotateImage(interpolationMode="linear"))

        # Shear image
        self.shearAction.triggered.connect(lambda: self.shearImage())
        
        # Equalize Image
        self.showHistogramAction.triggered.connect(lambda: self.currentTab.addHistogram())
        self.equalizeAction.triggered.connect(lambda: self.equalizeImage()) 

        self.addTabAction.triggered.connect(lambda: self.addNewTab())
        
        self.unsharpAction.triggered.connect(self.applyFilter)
        self.addSaltNoiseAction.triggered.connect(self.addNoise)
        self.medianFilterAction.triggered.connect(self.applyMedian)

        self.save.triggered.connect(self.saveImage)
        self.exitAction.triggered.connect(lambda: self.exit()) # When click on exit action
    
    def _connect(self):
        self._connectActions()
        # Tabs
        self.tabs.currentChanged.connect(self.setCurrentTab)
        self.tabs.tabCloseRequested.connect(self.closeCurrentTab)
        self.tabs.tabBarDoubleClicked.connect(self.tabOpenDoubleclick)

    def addNoise(self):
        self.currentTab.primaryViewer.addSaltAndPepper()
    
    def applyMedian(self):
        self.currentTab.primaryViewer.medianMask()

    def applyFilter(self):
        try:
            filterSize = int(self.sizeInput.text())
            factorSize = int(self.factorInput.text())

        except Exception as e:
            print(e)
            QMessageBox.critical(self , "Invalid size or factor" , "Please enter valid size or factor.")
            return

        if filterSize > 0:
            try:
                if filterSize % 2 == 0:
                    filterSize += 1

                self.currentTab.primaryViewer.unsharpMask(filterSize,factorSize)
            except Exception as e:
                print(e)
                QMessageBox.critical(self,"Error","Sorry, Error occurred.")
                return
                
            # self.setInfo(self.interpolationMode, self.widthOfImage, self.heightOfImage, self.sizeOfImage, self.depthOfImage, self.modeOfImage, self.modalityOfImage, self.nameOfPatient, self.ageOfPatient, self.bodyOfPatient)
        else:
            QMessageBox.critical(self , "Invalid size" , "Please enter valid size.")
        
    def tabOpenDoubleclick(self,i):
        # checking index i.e
        # No tab under the click
        if i == -1:
            # creating a new tab
            self.addNewTab()

    # Equalize
    def equalizeImage(self):
        self.currentTab.equalize()

    # Shear Image
    def shearImage(self):
        try:
            shearFactor = float(self.factorInput.text())
        except:
            QMessageBox.critical(self , "Invalid shearing factor" , "Please enter valid factor.")
            return
        
        if -90 < shearFactor < 90:
            try:
                self.currentTab.primaryViewer.shearImage(shearFactor)
            except:
                QMessageBox.critical(self,"Error","Sorry, Error occurred.")
                return
        else:
            QMessageBox.critical(self , "Invalid shearing factor" , "Shear angle should be between -90° and 90°.")

    # Zoom Image
    def zoomImage(self, interpolationMode):
        try:
            zoomingFactor = float(self.factorInput.text())
        except:
            QMessageBox.critical(self , "Invalid zooming factor" , "Please enter valid factor.")
            return
        
        if zoomingFactor > 0:
            try:
                self.widthOfImage, self.heightOfImage = self.currentTab.primaryViewer.zoomImage(zoomingFactor, interpolationMode)
            except:
                QMessageBox.critical(self,"Error","Sorry, Error occurred.")
                return
                
            if interpolationMode == "nearest":
                self.interpolationMode = "Zoom Nearest Neighbor"
            elif interpolationMode == "linear":
                self.interpolationMode = "Zoom Bilinear"

            self.setInfo(self.interpolationMode, self.widthOfImage, self.heightOfImage, self.sizeOfImage, self.depthOfImage, self.modeOfImage, self.modalityOfImage, self.nameOfPatient, self.ageOfPatient, self.bodyOfPatient)
        else:
            QMessageBox.critical(self , "Invalid zooming factor" , "Please enter valid factor.")

    # Rotate Image
    def rotateImage(self, interpolationMode):
        try:
            rotationAngle = float(self.factorInput.text())
        except:
            QMessageBox.critical(self , "Invalid Zooming Factor" , "Please Enter Valid Factor.")
            return

        
        self.widthOfImage, self.heightOfImage = self.currentTab.primaryViewer.rotateImage(rotationAngle, interpolationMode)

        direction = "Clockwise"
        if rotationAngle >= 0:
            direction = "Counterclockwise"
                
        if interpolationMode == "nearest":
            self.interpolationMode = "Rotate Nearest Neighbor"
        else :
            self.interpolationMode = "Rotate Bilinear"
        
        self.setInfo(self.interpolationMode, self.widthOfImage, self.heightOfImage, abs(rotationAngle), direction)
        
    # Open Image
    def browseImage(self):
        # Browse Function
        path, _ = QFileDialog.getOpenFileName(None, "Load Image File", filter="Custom files (*.bmp *.jpeg *.jpg *.dcm);;All files (*.*)")            
        self.fileExtension = path.split(".")[-1] # get ext.
        
        # If no image choosed
        if path == "":
            return

        try:
            data = self.currentTab.setImage(path, self.fileExtension)
        except:
            # Error
            appLogger.exception("Can't open the file !")
            QMessageBox.critical(self , "Corrupted image" , "Can't open the file !")
        else:
            self.currentTab.histogramViewer.drawHistogram(self.currentTab.primaryViewer.grayImage)

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

    def saveImage(self):
        output_file, _ = QFileDialog.getSaveFileName(self, 'Save image', None, 'jpeg files (.jpeg)')
        if output_file != '':
            if QFileInfo(output_file).suffix() == "" : output_file += '.jpeg'
        
        self.currentTab.primaryViewer.saveImage(output_file)

    # Control the tabs
    def setCurrentTab(self):
        self.currentTab = self.tabs.currentWidget()
    
    # Close tab
    def closeCurrentTab(self, i):
        # if there is only one tab
        if self.tabs.count() < 2:
            # do nothing
            return
 
        # else remove the tab
        self.tabs.removeTab(i)

    # Add new tab to list of tabs
    def addNewTab(self, title:str="Blank", color:str="black"):
        # Initilize new tab
        newTab = tabViewer(title, color)
        # Add tab to list of tabs
        self.tabs.addTab(newTab, title)
        # Return new tab
        return newTab
                
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