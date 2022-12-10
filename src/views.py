from math import ceil, log2
import cv2 as cv
import numpy as np

# Resources
from .rcIcon import *

# Importing sys package
import sys

# Import Classes
from .tabViewer import tabViewer

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
        self.images = []

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
        self.setWindowIcon(QIcon(":icon"))

        ### Setting title
        self.setWindowTitle("Image Processing Algorithms")

        ### UI contents
        self._createActions()
        self._createMenuBar()
        self._createToolBar("main")
        self._createToolBar("transformations")
        self._createToolBar("shapes")
        self._createToolBar("filters")
        self._createToolBar("inputs")
        self._createStatusBar()
        # Central area
        self._initUI()
        # Connect signals
        self._connect()
   
    ##########################################

    # Actions
    def _createActions(self):
        # Actions
        self.fileActions()
        self.imageActions()
        self.fourierActions()
        self.filtersActions()
        self.transformationsActions()
        self.operationsActions()
        self.constructionShapesActions()
        self.noisesActions()
        self.viewActions()
        self.helpActions()

    # File Actions
    def fileActions(self):
        # Add new tab Action
        self.addTabAction = QAction(QIcon(":add"), "&New...", self)
        self.addTabAction.setShortcut("Ctrl+N")
        self.addTabAction.setStatusTip('Add a new tab')

        # Open Action
        self.openAction = QAction(QIcon(":image"), "&Open Image...", self)
        self.openAction.setShortcut("Ctrl+O")
        self.openAction.setStatusTip('Open a new image')

        # Save the image Action
        self.saveAction = QAction(QIcon(":save"), "&Save image", self)
        self.saveAction.setShortcut("ctrl+S")
        self.saveAction.setStatusTip('Save the image')

        # Clear Action
        self.clearAction = QAction(QIcon(":clear"), "&Close Image", self)
        self.clearAction.setShortcut("Ctrl+C")
        self.clearAction.setStatusTip('Close the image')

        # Exit Action
        self.exitAction = QAction(QIcon(":exit"), "&Exit", self)
        self.exitAction.setShortcut("Ctrl+Q")
        self.exitAction.setStatusTip('Exit application')

    # Image Actions
    def imageActions(self):
        # Equalize the image
        self.equalizeAction = QAction(QIcon(":equalize"), "&Equalize", self)
        self.equalizeAction.setShortcut("ctrl+E")
        self.equalizeAction.setStatusTip('Equalize the image')

    # Edit Actions
    def fourierActions(self):
        # Log the magnitude of the image
        self.logMagnitudeAction = QAction(QIcon(":log"), "&Log magnitude", self)
        self.logMagnitudeAction.setShortcut("ctrl+L")
        self.logMagnitudeAction.setStatusTip('Log the image')

    # Filters Actions
    def filtersActions(self):
        # Unsharp masking Action
        self.unsharpMaskAction = QAction(QIcon(":box"), "&Un-sharp Mask", self)
        self.unsharpMaskAction.setShortcut("ctrl+U")
        self.unsharpMaskAction.setStatusTip('Create a Unsharp Masking')

        # box filter Action
        self.boxFilteringAction = QAction(QIcon(":blur"), "&Blur", self)
        self.boxFilteringAction.setShortcut("ctrl+j")
        self.boxFilteringAction.setStatusTip('Blur image in spatial domain')

        # box filter Action
        self.boxFilteringByFourierAction = QAction(QIcon(":blurFourier"), "&Blur using fourier", self)
        self.boxFilteringByFourierAction.setShortcut("ctrl+k")
        self.boxFilteringByFourierAction.setStatusTip('Blur image in frequency domain using fourier transform')

        # Median filter (50th percentile)
        self.medianFilterAction = QAction(QIcon(":median"), "&Median filter", self)
        self.medianFilterAction.setShortcut("ctrl+M")
        self.medianFilterAction.setStatusTip('Delete salt and pepper noise')

    # Transformation Actions
    def transformationsActions(self):
        # Zoom Nearest Neighbor Interpolation Action
        self.zoomNearestNeighborInterpolationAction = QAction(QIcon(":NNZoom"), "&Zoom by Nearest Neighbor", self)
        self.zoomNearestNeighborInterpolationAction.setShortcut("Ctrl+1")
        self.zoomNearestNeighborInterpolationAction.setStatusTip('Zoom in/out by Nearest Neighbor Interpolation method based on input')

        # Zoom Linear Interpolation Action
        self.zoomLinearInterpolationAction = QAction(QIcon(":LZoom"), "&Zoom by Linear", self)
        self.zoomLinearInterpolationAction.setShortcut("Ctrl+2")
        self.zoomLinearInterpolationAction.setStatusTip('Zoom in/out by Linear Interpolation method based on input')

        # Rotate the image by nearest neighbour interpolation
        self.rotateNearestAction = QAction(QIcon(":rotate"), "&Rotate by Nearest Neighbor", self)
        self.rotateNearestAction.setShortcut("ctrl+3")
        self.rotateNearestAction.setStatusTip('Rotate the image')

        # Rotate the image by linear interpolation
        self.rotateLinearAction = QAction(QIcon(":rotate"), "&Rotate by Linear", self)
        self.rotateLinearAction.setShortcut("ctrl+4")
        self.rotateLinearAction.setStatusTip('Rotate the image')

        # Shear the image horizontally
        self.shearActionHorizontal = QAction(QIcon(":shear"), "&Shear horizontally", self)
        self.shearActionHorizontal.setShortcut("ctrl+5")
        self.shearActionHorizontal.setStatusTip('Shear the image')

        # Shear the image horizontally
        self.shearActionVertical = QAction(QIcon(":shear"), "&Shear vertically", self)
        self.shearActionVertical.setShortcut("ctrl+6")
        self.shearActionVertical.setStatusTip('Shear the image')

    # Operations Actions
    def operationsActions(self):
        # Subtraction two image
        self.subtractionAction = QAction(QIcon(":subtraction"), "&Subtraction", self)
        self.subtractionAction.setShortcut("ctrl+D")
        self.subtractionAction.setStatusTip('Subtract two images')

        # Addition two image
        self.additionAction = QAction(QIcon(":addition"), "&Additions", self)
        self.additionAction.setShortcut("ctrl+A")
        self.additionAction.setStatusTip('Sum two images')

        # Add to the comparing list
        self.addToCompareListAction = QAction(QIcon(":compare"), "&Compare...", self)
        self.addToCompareListAction.setShortcut("Ctrl+B")
        self.addToCompareListAction.setStatusTip('Add to compare list')

    # Shapes Constructions Actions
    def constructionShapesActions(self):
        # Construct T image Action
        self.constructTAction = QAction(QIcon(":T"), "&Construct T", self)
        self.constructTAction.setShortcut("ctrl+T")
        self.constructTAction.setStatusTip('Construct an image with a T letter in the center')

        # Construct triangle image Action
        self.constructTriangleAction = QAction(QIcon(":triangle"), "&Construct Triangle", self)
        self.constructTriangleAction.setShortcut("ctrl+M")
        self.constructTriangleAction.setStatusTip('Construct an Triangle')
    
    # Noises Actions
    def noisesActions(self):
        # Add salt and pepper noise action
        self.addSaltPepperNoiseAction = QAction(QIcon(":salt"), "&Add salt and pepper", self)
        self.addSaltPepperNoiseAction.setShortcut("ctrl+P")
        self.addSaltPepperNoiseAction.setStatusTip('Add salt and pepper noise')
    
    # View Actions
    def viewActions(self):
        # Show histogram of the image
        self.showHistogramAction = QAction(QIcon(":histogram"), "&Histogram", self)
        self.showHistogramAction.setShortcut("ctrl+H")
        self.showHistogramAction.setStatusTip('Show the histogram')

        # Show histogram of the image
        self.showFourierAction = QAction(QIcon(":showFourier"), "&Fourier", self)
        self.showFourierAction.setShortcut("ctrl+F")
        self.showFourierAction.setStatusTip('Show the magnitude and phase')

    # Help Actions
    def helpActions(self):
        self.helpContentAction = QAction("&Help Content", self)
        self.helpContentAction.setStatusTip('Help')
        
        self.checkUpdatesAction = QAction("&Check For Updates", self)
        self.checkUpdatesAction.setStatusTip('Check Updates')
        
        self.aboutAction = QAction("&About", self)
        self.aboutAction.setStatusTip('About')

    ##########################################
    
    # Add separator
    def addSeparator(self, parent):
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
        fileMenu.addAction(self.addTabAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.openAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.saveAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.clearAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
    
        editMenu = QMenu("&Edit", self)
        editMenu.addAction(self.equalizeAction)
        editMenu.addSeparator()
        editMenu.addAction(self.logMagnitudeAction)

        viewMenu = QMenu("&View", self)
        viewMenu.addAction(self.showHistogramAction)
        viewMenu.addAction(self.showFourierAction)
        
        transformationMenu = QMenu("&Transformation", self)
        transformationMenu.addAction(self.zoomNearestNeighborInterpolationAction)
        transformationMenu.addAction(self.zoomLinearInterpolationAction)
        transformationMenu.addSeparator()
        transformationMenu.addAction(self.rotateNearestAction)
        transformationMenu.addAction(self.rotateLinearAction)
        transformationMenu.addSeparator()
        transformationMenu.addAction(self.shearActionHorizontal)
        transformationMenu.addAction(self.shearActionVertical)

        operationMenu = QMenu("&Operation", self)
        operationMenu.addAction(self.subtractionAction)
        operationMenu.addSeparator()
        operationMenu.addAction(self.addToCompareListAction)

        filterMenu = QMenu("&Filter", self)
        filterMenu.addAction(self.unsharpMaskAction)
        filterMenu.addAction(self.boxFilteringAction)
        filterMenu.addAction(self.boxFilteringByFourierAction)
        filterMenu.addAction(self.medianFilterAction)        

        shapeMenu = QMenu("&Shape", self)
        shapeMenu.addAction(self.constructTAction)
        shapeMenu.addAction(self.constructTriangleAction)

        noiseMenu = QMenu("&Noise", self)
        noiseMenu.addAction(self.addSaltPepperNoiseAction)

        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addSeparator()
        helpMenu.addAction(self.checkUpdatesAction)
        helpMenu.addSeparator()
        helpMenu.addAction(self.aboutAction)
        
        ## Append taps
        menuBar.addMenu(fileMenu)
        menuBar.addMenu(editMenu)
        menuBar.addMenu(transformationMenu)
        menuBar.addMenu(operationMenu)
        menuBar.addMenu(filterMenu)
        menuBar.addMenu(shapeMenu)
        menuBar.addMenu(noiseMenu)
        menuBar.addMenu(viewMenu)
        menuBar.addMenu(helpMenu)

    # Tool Bar
    def _createToolBar(self, type=""):
        self.toolBar = QToolBar("Tool Bar")
        
        if type == "main":
            # Using a title
            self.addToolBar(Qt.LeftToolBarArea, self.toolBar)  # type: ignore
            self.toolBar.addAction(self.openAction)
            
            self.toolBar.addAction(self.showHistogramAction)
            self.toolBar.addAction(self.showFourierAction)
            self.toolBar.addAction(self.equalizeAction)
            self.toolBar.addAction(self.logMagnitudeAction)
            self.toolBar.addAction(self.addToCompareListAction)
            self.toolBar.addAction(self.clearAction)       

        elif type == "transformations":
            self.addToolBar(Qt.TopToolBarArea,self.toolBar) # type: ignore
            self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon) # type: ignore
            
            self.toolBar.addAction(self.zoomNearestNeighborInterpolationAction)
            self.toolBar.addAction(self.zoomLinearInterpolationAction)
            self.toolBar.addAction(self.rotateNearestAction)
            self.toolBar.addAction(self.rotateLinearAction)
            self.toolBar.addAction(self.shearActionHorizontal)     

        elif type == "shapes":
            self.addToolBar(Qt.RightToolBarArea, self.toolBar)  # type: ignore
            
            self.toolBar.addAction(self.constructTAction)
            self.toolBar.addAction(self.constructTriangleAction)

        elif type == "filters":
            self.addToolBar(Qt.TopToolBarArea, self.toolBar)  # type: ignore
            
            self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon) # type: ignore
            self.toolBar.addAction(self.unsharpMaskAction) 
            self.toolBar.addAction(self.boxFilteringAction)
            self.toolBar.addAction(self.boxFilteringByFourierAction)
            self.toolBar.addAction(self.addSaltPepperNoiseAction)
            self.toolBar.addAction(self.medianFilterAction)
            self.toolBar.addAction(self.subtractionAction)
            
        elif type == "inputs":
            self.addToolBar(Qt.BottomToolBarArea, self.toolBar)  # type: ignore
            
            self.sizeInput = self.addInput("Filter Size")
            self.factorInput = self.addInput("Factor or Highboost")
            self.toolBar.addWidget(self.sizeInput)
            self.toolBar.addWidget(self.factorInput)
    
    # Context Menu Event
    def contextMenuEvent(self, event):
        # Creating a menu object with the central widget as parent
        menu = QMenu(self)
        # Populating the menu with actions
        menu.addAction(self.openAction)
        menu.addAction(self.saveAction)
        self.addSeparator(menu)
        menu.addAction(self.equalizeAction)
        menu.addAction(self.clearAction)
        self.addSeparator(menu)
        menu.addAction(self.constructTAction)
        self.addSeparator(menu)
        menu.addAction(self.helpContentAction)
        menu.addAction(self.checkUpdatesAction)
        menu.addAction(self.aboutAction)
        # Launching the menu
        menu.exec(event.globalPos())
    
    ##########################################

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

    # Status Bar
    def _createStatusBar(self):
        self.statusbar = self.statusBar()
        self.statusbar.setStyleSheet(f"""font-size:15px;
                                 padding: 4px;""")
        self.statusbar.showMessage("Ready", 3000)

        # Adding a permanent message
        self.statusbar.addPermanentWidget(QLabel("Image processing algorithms"))

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

    # Get Data
    def getAttr(self, variable, att):
        if hasattr(variable, att):
            # If attribute is found.
            return getattr(variable, att)
        else:
            # If attribute is not found.
            return "N/A"

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
        # File Actions
        self.addTabAction.triggered.connect(self.addNewTab)
        self.openAction.triggered.connect(self.browseImage) 
        self.clearAction.triggered.connect(self.clearImage) 

        "Transformation image"
        # Zoom image
        self.zoomNearestNeighborInterpolationAction.triggered.connect(lambda: self.zoomImage("nearest"))
        self.zoomLinearInterpolationAction.triggered.connect(lambda: self.zoomImage("linear"))
        # Rotate image
        self.rotateNearestAction.triggered.connect(lambda: self.rotateImage(mode="nearest"))
        self.rotateLinearAction.triggered.connect(lambda: self.rotateImage(mode="linear"))
        # Shear image
        self.shearActionHorizontal.triggered.connect(lambda: self.shearImage(mode="horizontal"))
        self.shearActionVertical.triggered.connect(lambda: self.shearImage(mode="vertical"))

        "Shapes construction"
        # Construct T
        self.constructTAction.triggered.connect(lambda: self.currentTab.drawT())
        # Construct triangle
        self.constructTriangleAction.triggered.connect(lambda: self.currentTab.drawTriangle())
        
        
        self.showHistogramAction.triggered.connect(lambda: self.currentTab.showHideHistogram())
        
        # Equalize Image
        self.equalizeAction.triggered.connect(lambda: self.equalizeImage()) 
        
        "Filters"
        self.unsharpMaskAction.triggered.connect(self.applyUnsharp)
        self.medianFilterAction.triggered.connect(self.applyMedian)
        self.boxFilteringAction.triggered.connect(lambda: self.applyBoxFilter("spatial"))
        self.boxFilteringByFourierAction.triggered.connect(lambda: self.applyBoxFilter("frequency"))

        self.addSaltPepperNoiseAction.triggered.connect(self.addNoise)

        self.subtractionAction.triggered.connect(lambda: self.subtractionTwoImage(self.images))
        # .triggered.connect(lambda: self.subtractionTwoImage(self.images))

        self.addToCompareListAction.triggered.connect(self.addToCompare)

        self.showFourierAction.triggered.connect(self.showFourier)
        self.logMagnitudeAction.triggered.connect(self.logMagnitude)

        self.saveAction.triggered.connect(self.saveImage)
        self.exitAction.triggered.connect(lambda: self.exit()) # When click on exit action
    
    def _connect(self):
        self._connectActions()
        # Tabs
        self.tabs.currentChanged.connect(self.setCurrentTab)
        self.tabs.tabCloseRequested.connect(self.closeCurrentTab)
        self.tabs.tabBarDoubleClicked.connect(self.tabOpenDoubleClick)

    # Add input
    def addInput(self, placeholderText):
        inputField = QLineEdit()
        inputField.setPlaceholderText(placeholderText)
        inputField.setStyleSheet("""border:1px solid #00d; 
                                            height:18px; 
                                            padding:2px; 
                                            border-radius:5px; 
                                            font-size:16px; 
                                            margin-right:5px""")
        return inputField

    # Get depth of image
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

    ##########################################
    #         """File Functions"""           #
    ##########################################

    # Open Image
    def browseImage(self):
        # Browse Function
        path, _ = QFileDialog.getOpenFileName(None, "Load Image File", filter="Custom files (*.bmp *.jpeg *.jpg *.dcm);;All files (*.*)")            
        self.fileExtension = path.split(".")[-1] # get ext.
        
        # If no image chosen
        if path == "":
            return

        try:
            data = self.currentTab.setImage(path, self.fileExtension)
        except:
            # Error
            appLogger.exception("Can't open the file !")
            QMessageBox.critical(self , "Corrupted image" , "Can't open the file !")
        else:
            self.updateImage()

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

    # Save Image
    def saveImage(self):
        output_file, _ = QFileDialog.getSaveFileName(self, 'Save image', None, 'jpeg files (.jpeg)')
        if output_file != '':
            if QFileInfo(output_file).suffix() == "" : output_file += '.jpeg'
        
        self.currentTab.primaryViewer.saveImage(output_file)

    # Clear image
    def clearImage(self):
        self.currentTab.primaryViewer.reset()
        self.currentTab.histogramViewer.reset()
        self.currentTab.magnitudeViewer.reset()
        self.currentTab.phaseViewer.reset() 

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

    ##########################################
    #         """Image Functions"""          #
    ##########################################

    # Equalize
    def equalizeImage(self):
        try:
            self.currentTab.equalize()
            self.updateImage()
        except:
            QMessageBox.critical(self,"Error","Sorry, Error occurred.")
            return

    ##########################################
    #         """Fourier Functions"""          #
    ##########################################

    # Log the magnitude of fourier transformed image
    def logMagnitude(self):
        self.updateImage(True)
        self.currentTab.primaryViewer.logImage()

    ##########################################
    #       """Filters Functions"""          #
    ##########################################
    
    # Apply box filter
    def applyBoxFilter(self, mode="spatial"):
        try:
            filterSize = int(self.sizeInput.text())
        except Exception as e:
            print(e)
            QMessageBox.critical(self , "Invalid size or factor" , "Please enter valid size or factor.")
            return

        if filterSize > 0:
            if filterSize % 2 == 0:
                filterSize += 1

            try:
                if mode == "frequency":
                    self.currentTab.primaryViewer.boxFilteringUsingFourier(filterSize)
                else:
                    self.currentTab.primaryViewer.boxFiltering(filterSize)
                self.updateImage()

            except Exception as e:
                print(e)
                QMessageBox.critical(self,"Error","Sorry, Error occurred.")
                return
        
    # Apply median masking
    def applyMedian(self): 
        try:
            filterSize = int(self.sizeInput.text())
        except Exception as e:
            print(e)
            QMessageBox.critical(self , "Invalid size or factor" , "Please enter valid size or factor.")
            return

        if filterSize > 0:
            if filterSize % 2 == 0:
                filterSize += 1
            try:
                self.currentTab.primaryViewer.medianMask(filterSize)
                self.updateImage()
            except Exception as e:
                print(e)
                QMessageBox.critical(self,"Error","Sorry, Error occurred.")
                return

    # Apply un-sharp masking
    def applyUnsharp(self):
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
                self.updateImage()
            except Exception as e:
                print(e)
                QMessageBox.critical(self,"Error","Sorry, Error occurred.")
                return
                
            # self.setInfo(self.interpolationMode, self.widthOfImage, self.heightOfImage, self.sizeOfImage, self.depthOfImage, self.modeOfImage, self.modalityOfImage, self.nameOfPatient, self.ageOfPatient, self.bodyOfPatient)
        else:
            QMessageBox.critical(self , "Invalid size" , "Please enter valid size.")
        
    ##########################################
    #     """Transformations Functions"""    #
    ##########################################

    # Shear Image
    def shearImage(self, mode="horizontal"):
        try:
            shearFactor = float(self.factorInput.text())
        except:
            QMessageBox.critical(self , "Invalid shearing factor" , "Please enter valid factor.")
            return
        
        if -90 < shearFactor < 90:
            try:
                self.currentTab.primaryViewer.shearImage(shearFactor, mode)
                self.updateImage()
            except:
                QMessageBox.critical(self,"Error","Sorry, Error occurred.")
                return
        else:
            QMessageBox.critical(self , "Invalid shearing factor" , "Shear angle should be between -90° and 90°.")

    # Zoom Image
    def zoomImage(self, mode="linear"):
        try:
            zoomingFactor = float(self.factorInput.text())
        except:
            QMessageBox.critical(self , "Invalid zooming factor" , "Please enter valid factor.")
            return
        
        if zoomingFactor > 0:
            try:
                self.widthOfImage, self.heightOfImage = self.currentTab.primaryViewer.zoomImage(zoomingFactor, mode)
                self.updateImage()
            except:
                QMessageBox.critical(self,"Error","Sorry, Error occurred.")
                return
                
            if mode == "nearest":
                self.interpolationMode = "Zoom Nearest Neighbor"
            elif mode == "linear":
                self.interpolationMode = "Zoom Bilinear"

            self.setInfo(self.interpolationMode, self.widthOfImage, self.heightOfImage, self.sizeOfImage, self.depthOfImage, self.modeOfImage, self.modalityOfImage, self.nameOfPatient, self.ageOfPatient, self.bodyOfPatient)
        else:
            QMessageBox.critical(self , "Invalid zooming factor" , "Please enter valid factor.")

    # Rotate Image
    def rotateImage(self, mode="linear"):
        try:
            rotationAngle = float(self.factorInput.text())
        except:
            QMessageBox.critical(self , "Invalid Zooming Factor" , "Please Enter Valid Factor.")
            return

        
        self.widthOfImage, self.heightOfImage = self.currentTab.primaryViewer.rotateImage(rotationAngle, mode)

        direction = "Clockwise"
        if rotationAngle >= 0:
            direction = "Counterclockwise"
                
        if mode == "nearest":
            self.interpolationMode = "Rotate Nearest Neighbor"
        else :
            self.interpolationMode = "Rotate Bilinear"
        
        self.setInfo(self.interpolationMode, self.widthOfImage, self.heightOfImage, abs(rotationAngle), direction)

    ##########################################
    #       """Operations Functions"""       #
    ##########################################
    
    # Add to comparing list
    def addToCompare(self):
        image = self.currentTab.primaryViewer.grayImage
        if len(self.images) > 1:
            self.images = []
        
        self.images.append(image)

    # get subtraction of images
    def subtractionTwoImage(self, images):
        if len(images) == 2:
            titleOfNewWindow = f"Subtraction of images"
            newTab = self.addNewTab(titleOfNewWindow,type="compare")
            newTab.primaryViewer.subtractionTwoImage(images[0], images[1])
            newTab.secondaryViewer.subtractionTwoImage(images[1], images[0])

    # get addition of images
    def additionTwoImage(self, images):
        if len(images) == 2:
            titleOfNewWindow = f"Addition of images"
            newTab = self.addNewTab(titleOfNewWindow)
            newTab.primaryViewer.additionTwoImage(images[0], images[1])

    ##########################################
    #    """Construct Shapes Functions"""    #
    ##########################################

    # Draw T shape
    def drawT(self):
        self.currentTab.primaryViewer.constructT("white")
        self.currentTab.histogramViewer.drawHistogram(self.primaryViewer.grayImage)

    # Draw triangle shape
    def drawTriangle(self):
        self.currentTab.primaryViewer.constructTriangle("white")
        self.currentTab.histogramViewer.drawHistogram(self.primaryViewer.grayImage)

    ##########################################
    #         """Noise Functions"""          #
    ##########################################

    # Add random noise to the image
    def addNoise(self):
        self.currentTab.primaryViewer.addSaltAndPepper()
        self.updateImage()

    ##########################################
    #         """View Functions"""           #
    ##########################################

    # Show magnitude and phase of image
    def showFourier(self):
        self.currentTab.showHideMagnitude()
        self.currentTab.showHidePhase()

    ##########################################
    
    # Update magnitude and phase and histogram for the image after every update
    def updateImage(self,log=False):
        if log:
            self.currentTab.magnitudeViewer.fourierTransform(self.currentTab.primaryViewer.grayImage,"magnitude",True)
            self.currentTab.phaseViewer.fourierTransform(self.currentTab.primaryViewer.grayImage,"phase",True)
        else:
            self.currentTab.magnitudeViewer.fourierTransform(self.currentTab.primaryViewer.grayImage,"magnitude")
            self.currentTab.phaseViewer.fourierTransform(self.currentTab.primaryViewer.grayImage,"phase")

        self.currentTab.histogramViewer.drawHistogram(self.currentTab.primaryViewer.grayImage)
    
    # Open new tap when double click
    def tabOpenDoubleClick(self,i):
        # checking index i.e
        # No tab under the click
        if i == -1:
            # creating a new tab
            self.addNewTab()

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
    def addNewTab(self, title:str="Blank", color:str="black", type="normal"):
        # Initialize new tab
        newTab = tabViewer(title, color, type)
        # Add tab to list of tabs
        self.tabs.addTab(newTab, title)
        # Return new tab
        return newTab