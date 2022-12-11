from math import ceil, log2
import cv2 as cv
import numpy as np

# Resources
from .rcIcon import *

# Importing sys package
import sys

# Import Classes
from .tabViewer import tabViewer
from .popup import popWindow
# Importing Qt widgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import matplotlib
import matplotlib.pyplot as plt

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

        self.setShortcuts()

    # File Actions
    def fileActions(self):
        # Add new tab Action
        self.addTabAction = QAction(QIcon(":add"), "&New...", self)
        self.addTabAction.setStatusTip('Add a new tab')

        # Open Action
        self.openAction = QAction(QIcon(":image"), "&Open Image...", self)
        self.openAction.setStatusTip('Open a new image')

        # Save the image Action
        self.saveAction = QAction(QIcon(":save"), "&Save image", self)
        self.saveAction.setStatusTip('Save the image')

        # Clear Action
        self.clearAction = QAction(QIcon(":clear"), "&Close Image", self)
        self.clearAction.setStatusTip('Close the image')

        # Exit Action
        self.exitAction = QAction(QIcon(":exit"), "&Exit", self)
        self.exitAction.setStatusTip('Exit application')

    # Image Actions
    def imageActions(self):
        # Equalize the image
        self.equalizeAction = QAction(QIcon(":equalize"), "&Equalize", self)
        self.equalizeAction.setStatusTip('Equalize the image')

        # Transform to binary
        self.binaryAction = QAction(QIcon(":binary"), "&Binary image", self)
        self.binaryAction.setStatusTip('Transform to binary image')

        # Negative image
        self.negativeAction = QAction(QIcon(":negative"), "&Negative image", self)
        self.negativeAction.setStatusTip('Negative image')

        # Log image
        self.logImageAction = QAction(QIcon(":log"), "&Log", self)
        self.logImageAction.setStatusTip('Log image')

        # Gamma image
        self.gammaAction = QAction(QIcon(":gamma"), "&Gamma Correction", self)
        self.gammaAction.setStatusTip('Gamma image')

    # Edit Actions
    def fourierActions(self):
        # Log the magnitude of the image
        self.logMagnitudeAction = QAction(QIcon(":log"), "&Log magnitude", self)
        self.logMagnitudeAction.setStatusTip('Log the image')

    # Filters Actions
    def filtersActions(self):
        # Unsharp masking
        self.unsharpMaskAction = QAction(QIcon(":box"), "&Un-sharp Mask", self)
        self.unsharpMaskAction.setShortcut("ctrl+U")
        self.unsharpMaskAction.setStatusTip('Create a Unsharp Masking')

        # Box filter
        self.boxFilteringAction = QAction(QIcon(":blur"), "&Blur", self)
        self.boxFilteringAction.setShortcut("ctrl+j")
        self.boxFilteringAction.setStatusTip('Blur image in spatial domain')

        # Box filter (fourier)
        self.boxFilteringByFourierAction = QAction(QIcon(":blurFourier"), "&Blur using fourier", self)
        self.boxFilteringByFourierAction.setShortcut("ctrl+k")
        self.boxFilteringByFourierAction.setStatusTip('Blur image in frequency domain using fourier transform')

        # Median filter (50th percentile)
        self.medianFilterAction = QAction(QIcon(":median"), "&Median filter", self)
        self.medianFilterAction.setStatusTip('Delete salt and pepper noise')

        # Band reject filter
        self.notchRejectFilterAction = QAction(QIcon(":bandReject"), "&band Reject", self)
        self.notchRejectFilterAction.setStatusTip('Apply band reject filter on image')
    
    # Transformation Actions
    def transformationsActions(self):
        # Zoom Nearest Neighbor Interpolation Action
        self.zoomNearestNeighborInterpolationAction = QAction(QIcon(":NNZoom"), "&Zoom by Nearest Neighbor", self)
        self.zoomNearestNeighborInterpolationAction.setStatusTip('Zoom in/out by Nearest Neighbor Interpolation method based on input')

        # Zoom Linear Interpolation Action
        self.zoomLinearInterpolationAction = QAction(QIcon(":LZoom"), "&Zoom by Linear", self)
        self.zoomLinearInterpolationAction.setStatusTip('Zoom in/out by Linear Interpolation method based on input')

        # Rotate the image by nearest neighbour interpolation
        self.rotateNearestAction = QAction(QIcon(":rotate"), "&Rotate by Nearest Neighbor", self)
        self.rotateNearestAction.setStatusTip('Rotate the image')

        # Rotate the image by linear interpolation
        self.rotateLinearAction = QAction(QIcon(":rotate"), "&Rotate by Linear", self)
        self.rotateLinearAction.setStatusTip('Rotate the image')

        # Shear the image horizontally
        self.shearActionHorizontal = QAction(QIcon(":shear"), "&Shear horizontally", self)
        self.shearActionHorizontal.setStatusTip('Shear the image horizontally')

        # Shear the image horizontally
        self.shearActionVertical = QAction(QIcon(":shear"), "&Shear vertically", self)
        self.shearActionVertical.setStatusTip('Shear the image vertically')

    # Operations Actions
    def operationsActions(self):
        """ Arithmetic Operations """
        # Subtract two image
        self.subtractionAction = QAction(QIcon(":subtraction"), "&Subtraction", self)
        self.subtractionAction.setStatusTip('Subtract two images')

        # Add two image
        self.additionAction = QAction(QIcon(":addition"), "&Addition", self)
        self.additionAction.setStatusTip('Sum two images')

        # Multiply two image
        self.multiplicationAction = QAction(QIcon(":multiplication"), "&Multiplication", self)
        self.multiplicationAction.setStatusTip('multiply two images')

        # Divide two image
        self.divisionAction = QAction(QIcon(":division"), "&division", self)
        self.divisionAction.setStatusTip('divide two images')

        """ Set Operations """
        # Complement
        self.complementAction = QAction(QIcon(":complement"), "&Complement", self)
        self.complementAction.setStatusTip('Complement operator')

        # Union
        self.unionAction = QAction(QIcon(":union"), "&Union", self)
        self.unionAction.setStatusTip('Union operator')

        # Intersect
        self.intersectAction = QAction(QIcon(":intersect"), "&Intersect", self)
        self.intersectAction.setStatusTip('Intersect operator')

        """ Logical Operations """
        # Not
        self.notAction = QAction(QIcon(":not"), "&Not", self)
        self.notAction.setStatusTip('Not operator')

        # And
        self.andAction = QAction(QIcon(":and"), "&And", self)
        self.andAction.setStatusTip('And operator')

        # Nand
        self.nandAction = QAction(QIcon(":nand"), "&Nand", self)
        self.nandAction.setStatusTip('Nand operator')

        # Or
        self.orAction = QAction(QIcon(":or"), "&Or", self)
        self.orAction.setStatusTip('Or operator')

        # Nor
        self.norAction = QAction(QIcon(":nor"), "&Nor", self)
        self.norAction.setStatusTip('Nor operator')

        # Xor
        self.xorAction = QAction(QIcon(":xor"), "&Xor", self)
        self.xorAction.setStatusTip('Xor operator')

        # Xnor
        self.xnorAction = QAction(QIcon(":xnor"), "&Xnor", self)
        self.xnorAction.setStatusTip('Xnor operator')

        # Add to the comparing list
        self.addToCompareListAction = QAction(QIcon(":compare"), "&Compare...", self)
        self.addToCompareListAction.setStatusTip('Add to compare list')

    # Shapes Constructions Actions
    def constructionShapesActions(self):
        # Construct T image Action
        self.constructTAction = QAction(QIcon(":T"), "&Construct T", self)
        self.constructTAction.setStatusTip('Construct an image with a T letter in the center')

        # Construct triangle image Action
        self.constructTriangleAction = QAction(QIcon(":triangle"), "&Construct Triangle", self)
        self.constructTriangleAction.setStatusTip('Construct an Triangle')
    
    # Noises Actions
    def noisesActions(self):
        # Add salt and pepper noise action
        self.addSaltPepperNoiseAction = QAction(QIcon(":salt"), "&Add salt and pepper", self)
        self.addSaltPepperNoiseAction.setStatusTip('Add salt and pepper noise')
    
    # View Actions
    def viewActions(self):
        # Show histogram of the image
        self.showHistogramAction = QAction(QIcon(":histogram"), "&Histogram", self)
        self.showHistogramAction.setStatusTip('Show the histogram')

        # Show histogram of the image
        self.showFourierAction = QAction(QIcon(":showFourier"), "&Fourier", self)
        self.showFourierAction.setStatusTip('Show the magnitude and phase')

    # Help Actions
    def helpActions(self):
        self.helpContentAction = QAction("&Help Content", self)
        self.helpContentAction.setStatusTip('Help')
        
        self.checkUpdatesAction = QAction("&Check For Updates", self)
        self.checkUpdatesAction.setStatusTip('Check Updates')
        
        self.aboutAction = QAction("&About", self)
        self.aboutAction.setStatusTip('About')

    # Shortcuts
    def setShortcuts(self):
        "File Actions"
        self.addTabAction.setShortcut("Ctrl+N")
        self.openAction.setShortcut("Ctrl+O")
        self.saveAction.setShortcut("ctrl+S")
        self.clearAction.setShortcut("Ctrl+C")
        self.exitAction.setShortcut("Ctrl+Q")

        "Image Actions"
        self.equalizeAction.setShortcut("ctrl+E")
        self.binaryAction.setShortcut("ctrl+G")
        self.negativeAction.setShortcut("ctrl+n")
        self.logImageAction.setShortcut("ctrl+l")
        self.gammaAction.setShortcut("ctrl+g")
        
        "Fourier Actions"
        self.logMagnitudeAction.setShortcut("ctrl+L")

        "Filters Actions"
        self.unsharpMaskAction.setShortcut("ctrl+U")
        self.boxFilteringAction.setShortcut("ctrl+j")
        self.boxFilteringByFourierAction.setShortcut("ctrl+k")
        self.medianFilterAction.setShortcut("ctrl+M")
        self.notchRejectFilterAction.setShortcut("ctrl+B")
        
        "Transformations Actions"
        self.zoomNearestNeighborInterpolationAction.setShortcut("Ctrl+1")
        self.zoomLinearInterpolationAction.setShortcut("Ctrl+2")
        self.rotateNearestAction.setShortcut("ctrl+3")
        self.rotateLinearAction.setShortcut("ctrl+4")
        self.shearActionHorizontal.setShortcut("ctrl+5")
        self.shearActionVertical.setShortcut("ctrl+6")

        "Operations Actions"
        self.subtractionAction.setShortcut("ctrl+D")
        self.additionAction.setShortcut("ctrl+A")
        self.multiplicationAction.setShortcut("ctrl+*")
        self.divisionAction.setShortcut("ctrl+/")

        self.complementAction.setShortcut("ctrl+c")
        self.unionAction.setShortcut("ctrl+u")
        self.intersectAction.setShortcut("ctrl+i")

        self.notAction.setShortcut("ctrl+n")
        self.andAction.setShortcut("ctrl+d")
        self.nandAction.setShortcut("ctrl+d")
        self.orAction.setShortcut("ctrl+r")
        self.norAction.setShortcut("ctrl+[")
        self.xorAction.setShortcut("ctrl+x")
        self.xnorAction.setShortcut("ctrl+z")

        self.addToCompareListAction.setShortcut("Ctrl+B")

        "Construction Shapes Actions"
        self.constructTAction.setShortcut("ctrl+T")
        self.constructTriangleAction.setShortcut("ctrl+M")

        "Noises Actions"
        self.addSaltPepperNoiseAction.setShortcut("ctrl+P")

        "View Actions"
        self.showHistogramAction.setShortcut("ctrl+H")
        self.showFourierAction.setShortcut("ctrl+F")

        "HelpActions"
        self.helpContentAction.setShortcut("alt+H")
        self.checkUpdatesAction.setShortcut("alt+Z")
        self.aboutAction.setShortcut("alt+X") 

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

        """File"""
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

        """Edit"""
        editMenu = QMenu("&Edit", self)
        editMenu.addAction(self.equalizeAction)
        editMenu.addSeparator()
        editMenu.addAction(self.logMagnitudeAction)

        """View"""
        viewMenu = QMenu("&View", self)
        viewMenu.addAction(self.showHistogramAction)
        viewMenu.addAction(self.showFourierAction)
        
        """Transformation"""
        transformationMenu = QMenu("&Transformation", self)
        transformationMenu.addAction(self.zoomNearestNeighborInterpolationAction)
        transformationMenu.addAction(self.zoomLinearInterpolationAction)
        transformationMenu.addSeparator()
        transformationMenu.addAction(self.rotateNearestAction)
        transformationMenu.addAction(self.rotateLinearAction)
        transformationMenu.addSeparator()
        transformationMenu.addAction(self.shearActionHorizontal)
        transformationMenu.addAction(self.shearActionVertical)

        """Operation"""
        operationMenu = QMenu("&Operation", self)
        operationMenu.addAction(self.subtractionAction)
        operationMenu.addSeparator()
        operationMenu.addAction(self.addToCompareListAction)

        """Filter"""
        filterMenu = QMenu("&Filter", self)
        filterMenu.addAction(self.unsharpMaskAction)
        filterMenu.addAction(self.boxFilteringAction)
        filterMenu.addAction(self.boxFilteringByFourierAction)
        filterMenu.addAction(self.medianFilterAction)
        filterMenu.addAction(self.notchRejectFilterAction) 

        """Shape"""
        shapeMenu = QMenu("&Shape", self)
        shapeMenu.addAction(self.constructTAction)
        shapeMenu.addAction(self.constructTriangleAction)

        """Noise"""
        noiseMenu = QMenu("&Noise", self)
        noiseMenu.addAction(self.addSaltPepperNoiseAction)

        """Help"""
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
        info = dict() # Initialize the dicom
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
        " File "
        self.addTabAction.triggered.connect(self.addNewTab)
        self.openAction.triggered.connect(self.browseImage) 
        self.clearAction.triggered.connect(self.clearImage) 
        self.saveAction.triggered.connect(self.saveImage)
        self.exitAction.triggered.connect(self.exit)

        " Transformation image "
        # Zoom image
        self.zoomNearestNeighborInterpolationAction.triggered.connect(lambda: self.zoomImage("nearest"))
        self.zoomLinearInterpolationAction.triggered.connect(lambda: self.zoomImage("linear"))
        # Rotate image
        self.rotateNearestAction.triggered.connect(lambda: self.rotateImage(mode="nearest"))
        self.rotateLinearAction.triggered.connect(lambda: self.rotateImage(mode="linear"))
        # Shear image
        self.shearActionHorizontal.triggered.connect(lambda: self.shearImage(mode="horizontal"))
        self.shearActionVertical.triggered.connect(lambda: self.shearImage(mode="vertical"))

        " Shapes construction "
        # Construct T
        self.constructTAction.triggered.connect(lambda: self.drawT())
        # Construct triangle
        self.constructTriangleAction.triggered.connect(lambda: self.drawTriangle())
        
        " View "
        self.showHistogramAction.triggered.connect(lambda: self.currentTab.showHideHistogram())
        self.showFourierAction.triggered.connect(self.showFourier)
        
        # Equalize Image
        self.equalizeAction.triggered.connect(lambda: self.equalizeImage()) 
        
        " Filters "
        self.unsharpMaskAction.triggered.connect(self.applyUnsharp)
        self.medianFilterAction.triggered.connect(self.applyMedian)
        self.boxFilteringAction.triggered.connect(lambda: self.applyBoxFilter("spatial"))
        self.boxFilteringByFourierAction.triggered.connect(lambda: self.applyBoxFilter("frequency"))
        self.notchRejectFilterAction.triggered.connect(self.notchRejectFilter)
        
        " Noise "
        self.addSaltPepperNoiseAction.triggered.connect(self.addNoise)

        " Operation "
        self.subtractionAction.triggered.connect(lambda: self.subtractionTwoImage(self.images))
        self.addToCompareListAction.triggered.connect(self.addToCompare)
        
        " Fourier "
        self.logMagnitudeAction.triggered.connect(self.logMagnitude)
    
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
    
    def set_plot_title(self, title, fs = 16):
       plt.title(title, fontsize = fs)

    def notchRejectFilter(self):
        image = self.currentTab.primaryViewer.getImage()

        if len(image) != 0:

            requirements = {
                "Number of points":"",
                "Radius":""
            }

            inputWindow = popWindow("Notch Reject Filter", requirements)
            inputWindow.exec_()
            
            output = inputWindow.getValues()
            if output != None:
                n = int(output.get("Number of points"))
                frequency = int(output.get("Radius"))
            else:
                return

            plt.clf()

            spectrum = self.currentTab.primaryViewer.fourierTransform(image, draw=False)            
            magnitudeSpectrum = self.currentTab.primaryViewer.fourierTransform(image, mode="magnitude", log=True, draw=False)
            
            plt.imshow(magnitudeSpectrum, cmap = "gray")
            
            self.set_plot_title("Click on image to choose points. (Press any key to Start)")
            plt.waitforbuttonpress()
            self.set_plot_title(f'Select {n} points with mouse click')
            
            points = np.asarray(plt.ginput(n, timeout = -1))
            plt.close()

            self.currentTab.primaryViewer.notchRejectFilters(spectrum, points, frequency)
            
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
        print(len(self.images))
        

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
        self.updateImage()

    # Draw triangle shape
    def drawTriangle(self):
        self.currentTab.primaryViewer.constructTriangle("white")
        self.updateImage()

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