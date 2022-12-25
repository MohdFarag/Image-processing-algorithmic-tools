import numpy as np

# Resources
from .rcIcon import *

# Importing sys package
import sys

# Import Classes
from .tabViewer import tabViewer
from .popup import popWindow
from .utilities import *
from .style import *


# Importing Qt widgets
try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtGui import *
    from PyQt6.QtCore import *
except ImportError:
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Importing Logging
from .log import appLogger


# Window class
class MainWindow(QMainWindow):
    """Main Window"""
    def __init__(self, *args, **kwargs):
        """Initializer."""
        super(MainWindow, self).__init__(*args, **kwargs)

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
        
        ### Setting title
        self.setWindowTitle("Image Processing Algorithms")

        ### Setting Icon
        self.setWindowIcon(QIcon(":icon"))
        self.setMinimumSize(1000,600)

        ### UI contents
        self._createActions()
        self._createMenuBar()
        self._createToolBar("main")
        self._createToolBar("transformations")
        self._createToolBar("shapes")
        self._createToolBar("operation")
        self._createToolBar("inputs")
        self._createStatusBar()
        # Central area
        self._initUI()
        # Connect signals
        self._connect()
   
    ##########################################
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
        self.morphologicalActions()
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

        # Contrast Stretching
        self.contrastStretchingAction = QAction(QIcon(":contrastStretching"), "&Contrast Stretching", self)
        self.contrastStretchingAction.setStatusTip('Contrast Stretching')

        # Intensity level slicing
        self.intensityLevelSlicingAction = QAction(QIcon(":intensityLevelSlicing"), "&Intensity Level Slicing", self)
        self.intensityLevelSlicingAction.setStatusTip('Intensity Level Slicing')

        # Bit plane slicing
        self.bitPlaneSlicingAction = QAction(QIcon(":bitSlicing"), "&Bit plane slicing", self)
        self.bitPlaneSlicingAction.setStatusTip('Bit plane slicing')

        # Set region of interest
        self.setROIAction = QAction(QIcon(":ROI"), "&ROI", self)
        self.setROIAction.setStatusTip('Draw ROI')

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
        self.boxFilteringByFourierAction = QAction(QIcon(":blurFourier"), "&Blur in frequency domain", self)
        self.boxFilteringByFourierAction.setShortcut("ctrl+k")
        self.boxFilteringByFourierAction.setStatusTip('Blur image in frequency domain using fourier transform')

        # Median filter (50th percentile)
        self.medianFilterAction = QAction(QIcon(":median"), "&Median filter", self)
        self.medianFilterAction.setStatusTip('Delete salt and pepper noise')

        # Notch reject filter
        self.notchRejectFilterAction = QAction(QIcon(":bandReject"), "&Notch Reject", self)
        self.notchRejectFilterAction.setStatusTip('Apply notch reject filter on image')
    
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
        self.constructTAction = QAction(QIcon(":T"), "&Letter T", self)
        self.constructTAction.setStatusTip('Construct an image with a T letter in the center')

        # Construct triangle image Action
        self.constructTriangleAction = QAction(QIcon(":triangle"), "&Triangle", self)
        self.constructTriangleAction.setStatusTip('Construct an Triangle')

        # Construct triangle image Action
        self.constructCircleBoxAction = QAction(QIcon(":circle"), "&Circle", self)
        self.constructCircleBoxAction.setStatusTip('Construct an Circle in gray box')
    
        # Construct square image Action
        self.constructSquareBoxAction = QAction(QIcon(":square"), "&Square", self)
        self.constructSquareBoxAction.setStatusTip('Construct an square')

        # Construct background image Action
        self.constructBackgroundAction = QAction(QIcon(":background"), "&Background", self)
        self.constructBackgroundAction.setStatusTip('Construct an background with one color')
            
        # Construct Schepp-Logan phantom Action
        self.constructPhantomAction = QAction(QIcon(":phantom"), "&Schepp-Logan phantom", self)
        self.constructPhantomAction.setStatusTip('Construct an Schepp-Logan phantom')
    
    # Noises Actions
    def noisesActions(self):
        # Add gaussian noise action
        self.addUniformNoiseAction = QAction(QIcon(":uniform"), "&Uniform Noise", self)
        self.addUniformNoiseAction.setStatusTip('Add uniform noise')

        # Add gaussian noise action
        self.addGaussianNoiseAction = QAction(QIcon(":gaussian"), "&Gaussian Noise", self)
        self.addGaussianNoiseAction.setStatusTip('Add gaussian noise')

        # Add rayleigh noise action
        self.addRayleighNoiseAction = QAction(QIcon(":Rayleigh"), "&Rayleigh Noise", self)
        self.addRayleighNoiseAction.setStatusTip('Add rayleigh noise')

        # Add erlang noise action
        self.addErlangNoiseAction = QAction(QIcon(":Erlang"), "&Erlang Noise", self)
        self.addErlangNoiseAction.setStatusTip('Add erlang noise')

        # Add gaussian noise action
        self.addExponentialNoiseAction = QAction(QIcon(":Exponential"), "&Exponential Noise", self)
        self.addExponentialNoiseAction.setStatusTip('Add exponential noise')

        # Add salt noise action
        self.addSaltNoiseAction = QAction(QIcon(":salt"), "&Salt Noise", self)
        self.addSaltNoiseAction.setStatusTip('Add salt noise')

        # Add pepper noise action
        self.addPepperNoiseAction = QAction(QIcon(":pepper"), "&Pepper Noise", self)
        self.addPepperNoiseAction.setStatusTip('Add pepper noise')

        # Add salt and pepper noise action
        self.addSaltPepperNoiseAction = QAction(QIcon(":saltPepper"), "&Salt and Pepper Noise", self)
        self.addSaltPepperNoiseAction.setStatusTip('Add salt and pepper noise')

    # Morphological Actions
    def morphologicalActions(self):
        # Add erosion action
        self.erosionAction = QAction(QIcon(":erosion"), "&Erosion", self)
        self.erosionAction.setStatusTip('Erosion')

        # Add dilation action
        self.dilationAction = QAction(QIcon(":dilation"), "&Dilation", self)
        self.dilationAction.setStatusTip('Dilation')

        # Add opening action
        self.openingAction = QAction(QIcon(":opening"), "&Opening", self)
        self.openingAction.setStatusTip('Erosion followed by dilation')

        # Add closing action
        self.closingAction = QAction(QIcon(":closing"), "&Closing", self)
        self.closingAction.setStatusTip('Dilation followed by erosion')

        # Add closing action
        self.removeNoiseAction = QAction(QIcon(":removeNoise"), "&Remove Noise", self)
        self.removeNoiseAction.setStatusTip('Remove noise')

    # View Actions
    def viewActions(self):
        # Show histogram of the image
        self.showHistogramAction = QAction(QIcon(":histogram"), "&Histogram", self)
        self.showHistogramAction.setStatusTip('Show the histogram')

        # Show histogram of the image
        self.showFourierAction = QAction(QIcon(":showFourier"), "&Fourier", self)
        self.showFourierAction.setStatusTip('Show the magnitude and phase')

        # Show sinogram of the image
        self.showSinogramAction = QAction(QIcon(":sinogram"), "&Sinogram", self)
        self.showSinogramAction.setStatusTip('Show the sinogram')

        # Show histogram of the image
        self.showLaminogramAction = QAction(QIcon(":laminogram"), "&Laminogram", self)
        self.showLaminogramAction.setStatusTip('Show the laminogram')

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
        editMenu.addAction(self.logMagnitudeAction)

        """Image"""
        imageMenu = QMenu("&Image", self)
        imageMenu.addAction(self.equalizeAction)
        imageMenu.addSeparator()
        imageMenu.addAction(self.binaryAction)
        imageMenu.addAction(self.negativeAction)
        imageMenu.addAction(self.logImageAction)
        imageMenu.addAction(self.gammaAction)
        imageMenu.addSeparator()
        imageMenu.addAction(self.contrastStretchingAction)
        imageMenu.addAction(self.intensityLevelSlicingAction)
        imageMenu.addAction(self.bitPlaneSlicingAction)
        imageMenu.addSeparator()
        imageMenu.addAction(self.setROIAction)

        """View"""
        viewMenu = QMenu("&View", self)
        viewMenu.addAction(self.showHistogramAction)
        viewMenu.addAction(self.showFourierAction)
        viewMenu.addAction(self.showSinogramAction)
        viewMenu.addAction(self.showLaminogramAction)
        
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
        operationMenu.addAction(self.additionAction)
        operationMenu.addAction(self.subtractionAction)
        operationMenu.addAction(self.multiplicationAction)
        operationMenu.addAction(self.divisionAction)
        operationMenu.addSeparator()
        operationMenu.addAction(self.unionAction)
        operationMenu.addAction(self.intersectAction)
        operationMenu.addAction(self.complementAction)
        operationMenu.addSeparator()
        operationMenu.addAction(self.notAction)
        operationMenu.addAction(self.andAction)
        operationMenu.addAction(self.nandAction)
        operationMenu.addAction(self.orAction)
        operationMenu.addAction(self.norAction)
        operationMenu.addAction(self.xorAction)
        operationMenu.addAction(self.xnorAction)
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
        shapeMenu.addAction(self.constructCircleBoxAction)
        shapeMenu.addAction(self.constructSquareBoxAction)
        shapeMenu.addAction(self.constructPhantomAction)
        shapeMenu.addAction(self.constructBackgroundAction)

        """Noise"""
        noiseMenu = QMenu("&Noise", self)
        noiseMenu.addAction(self.addUniformNoiseAction)
        noiseMenu.addAction(self.addGaussianNoiseAction)
        noiseMenu.addAction(self.addRayleighNoiseAction)
        noiseMenu.addAction(self.addErlangNoiseAction)
        noiseMenu.addAction(self.addExponentialNoiseAction)
        noiseMenu.addSeparator()
        noiseMenu.addAction(self.addSaltPepperNoiseAction)
        noiseMenu.addAction(self.addSaltNoiseAction)
        noiseMenu.addAction(self.addPepperNoiseAction)

        """Morphological"""
        morphologicalMenu = QMenu("&Morphological", self)
        morphologicalMenu.addAction(self.erosionAction)
        morphologicalMenu.addAction(self.dilationAction)
        morphologicalMenu.addSeparator()
        morphologicalMenu.addAction(self.openingAction)
        morphologicalMenu.addAction(self.closingAction)
        morphologicalMenu.addAction(self.removeNoiseAction)
        
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
        menuBar.addMenu(imageMenu)
        menuBar.addMenu(transformationMenu)
        menuBar.addMenu(operationMenu)
        menuBar.addMenu(filterMenu)
        menuBar.addMenu(shapeMenu)
        menuBar.addMenu(noiseMenu)
        menuBar.addMenu(morphologicalMenu)
        menuBar.addMenu(viewMenu)
        menuBar.addMenu(helpMenu)

    # Tool Bar
    def _createToolBar(self, type=""):
        self.toolBar = QToolBar("Toolbar")
        
        if type == "main":
            # Using a title
            self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.toolBar)  # type: ignore
            self.toolBar.addAction(self.openAction)
            
            self.toolBar.addAction(self.showHistogramAction)
            self.toolBar.addAction(self.showFourierAction)
            
            self.toolBar.addAction(self.showSinogramAction)
            self.toolBar.addAction(self.showLaminogramAction)

            self.toolBar.addAction(self.addToCompareListAction)
            self.toolBar.addAction(self.clearAction)       

        elif type == "transformations":
            self.addToolBar(Qt.ToolBarArea.TopToolBarArea,self.toolBar) # type: ignore
            self.toolBar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon) # type: ignore
            
            self.toolBar.addAction(self.zoomNearestNeighborInterpolationAction)
            self.toolBar.addAction(self.zoomLinearInterpolationAction)
            self.toolBar.addAction(self.rotateNearestAction)
            self.toolBar.addAction(self.rotateLinearAction)
            self.toolBar.addAction(self.shearActionHorizontal)     
            self.toolBar.addAction(self.shearActionVertical)     

        elif type == "shapes":
            self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.toolBar)  # type: ignore
            
            self.toolBar.addAction(self.constructTAction)
            self.toolBar.addAction(self.constructCircleBoxAction)
            self.toolBar.addAction(self.constructSquareBoxAction)
            self.toolBar.addAction(self.constructPhantomAction)

        elif type == "operation":
            self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.toolBar)  # type: ignore
            
            self.toolBar.addAction(self.additionAction)
            self.toolBar.addAction(self.subtractionAction)

    # Context Menu Event
    def contextMenuEvent(self, event):
        # Creating a menu object with the central widget as parent
        menu = QMenu(self)
        # Populating the menu with actions
        menu.addAction(self.openAction)
        menu.addAction(self.saveAction)
        self.addSeparator(menu)
        menu.addAction(self.clearAction)
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
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dockInfo)

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
        elif ext == "ROI":
            info = {
                    "Mean":width, 
                    "Variance":height,
                    "Std":size
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
        self.addTabAction.triggered.connect(lambda: self.addNewTab("Blank"))
        self.openAction.triggered.connect(lambda: self.baseBehavior(self.browseImage)) 
        self.clearAction.triggered.connect(lambda: self.baseBehavior(self.clearImage)) 
        self.saveAction.triggered.connect(self.saveImage)
        self.exitAction.triggered.connect(self.exit)

        " Fourier "
        self.logMagnitudeAction.triggered.connect(lambda: self.updateImage())

        "Image"
        self.equalizeAction.triggered.connect(lambda: self.baseBehavior(self.currentTab.equalize))

        self.binaryAction.triggered.connect(lambda: self.baseBehavior(self.binaryImage))
        self.negativeAction.triggered.connect(lambda: self.baseBehavior(self.negativeImage))
        self.logImageAction.triggered.connect(lambda: self.baseBehavior(self.logImage))
        self.gammaAction.triggered.connect(lambda: self.baseBehavior(self.gammaImage))

        self.contrastStretchingAction.triggered.connect(lambda: self.baseBehavior(self.contrastStretching))
        self.intensityLevelSlicingAction.triggered.connect(lambda: self.baseBehavior(self.intensityLevelSlicing))
        self.bitPlaneSlicingAction.triggered.connect(self.bitPlaneSlicing)

        self.setROIAction.triggered.connect(self.getROI)

        " Transformation image "
        # Zoom image
        self.zoomNearestNeighborInterpolationAction.triggered.connect(lambda: self.baseBehavior(self.zoomImage,"nearest"))
        self.zoomLinearInterpolationAction.triggered.connect(lambda: self.baseBehavior(self.zoomImage,"bilinear"))
        # Rotate image
        self.rotateNearestAction.triggered.connect(lambda: self.baseBehavior(self.rotateImage,"nearest"))
        self.rotateLinearAction.triggered.connect(lambda: self.baseBehavior(self.rotateImage,"bilinear"))
        # Shear image
        self.shearActionHorizontal.triggered.connect(lambda: self.baseBehavior(self.shearImage,"horizontal"))
        self.shearActionVertical.triggered.connect(lambda: self.baseBehavior(self.shearImage,"vertical"))

        " Shapes construction "
        # Construct T
        self.constructTAction.triggered.connect(lambda: self.baseBehavior(self.drawShape,"T"))
        # Construct triangle
        self.constructTriangleAction.triggered.connect(lambda: self.baseBehavior(self.drawShape,"triangle"))
        # Construct circle
        self.constructCircleBoxAction.triggered.connect(lambda: self.baseBehavior(self.drawShape,"circle"))
        # Construct square
        self.constructSquareBoxAction.triggered.connect(lambda: self.baseBehavior(self.drawShape,"square"))
        # Construct background
        self.constructBackgroundAction.triggered.connect(lambda: self.baseBehavior(self.drawShape))
        # Construct phantom
        self.constructPhantomAction.triggered.connect(lambda: self.baseBehavior(self.drawShape,"phantom"))
        
        " Filters "
        self.unsharpMaskAction.triggered.connect(lambda: self.baseBehavior(self.applyUnsharp))
        self.medianFilterAction.triggered.connect(lambda: self.baseBehavior(self.applyMedian))
        self.boxFilteringAction.triggered.connect(lambda: self.baseBehavior(self.applyBoxFilter, "spatial"))
        self.boxFilteringByFourierAction.triggered.connect(lambda: self.baseBehavior(self.applyBoxFilter, "frequency"))
        self.notchRejectFilterAction.triggered.connect(lambda: self.baseBehavior(self.notchRejectFilter))
        
        " Noise "
        self.addUniformNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "uniform"))
        self.addGaussianNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "gaussian"))
        self.addRayleighNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "rayleigh"))
        self.addErlangNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "erlang"))
        self.addExponentialNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "exponential"))
        self.addSaltNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "salt"))
        self.addPepperNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "pepper"))
        self.addSaltPepperNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "salt and pepper"))

        " Operation "
        # Arithmetic Operations
        self.subtractionAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "subtract", self.images))
        self.additionAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "add", self.images))
        self.multiplicationAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "multiply", self.images))
        self.divisionAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "divide", self.images))
        # Set Operations
        self.complementAction.triggered.connect(lambda: self.baseBehavior(self.operationOneImage, "complement"))
        self.unionAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "union", self.images))
        self.intersectAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "intersect", self.images))
        # Logical Operations
        self.notAction.triggered.connect(lambda: self.baseBehavior(self.operationOneImage, "not"))
        self.andAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "and", self.images))
        self.nandAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "nand", self.images))        
        self.orAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "or", self.images))
        self.norAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "nor", self.images))
        self.xorAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "xor", self.images))        
        self.xnorAction.triggered.connect(lambda: self.baseBehavior(self.operationTwoImage, "xnor", self.images))        

        self.addToCompareListAction.triggered.connect(self.addToCompare)

        " Morphological "
        self.erosionAction.triggered.connect(lambda: self.currentTab.primaryViewer.morphologicalActions("erosion"))
        self.dilationAction.triggered.connect(lambda: self.currentTab.primaryViewer.morphologicalActions("dilation"))
        self.openingAction.triggered.connect(lambda: self.currentTab.primaryViewer.morphologicalActions("opening"))
        self.closingAction.triggered.connect(lambda: self.currentTab.primaryViewer.morphologicalActions("closing"))
        self.removeNoiseAction.triggered.connect(lambda: self.currentTab.primaryViewer.morphologicalActions("noise"))

        " View "
        self.showHistogramAction.triggered.connect(lambda: self.currentTab.showHideHistogram())
        self.showFourierAction.triggered.connect(lambda: self.currentTab.showHideFourier())
        self.showSinogramAction.triggered.connect(self.sinogram)
        self.showLaminogramAction.triggered.connect(self.laminogram)

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

    ##########################################
    #         """File Functions"""           #
    ##########################################

    # Open Image
    def browseImage(self):
        # Browse Function
        path, _ = QFileDialog.getOpenFileName(self, "Load Image File", directory="./src/assets/testInputs/", filter="Custom files (*.bmp *.png *.jpeg *.jpg *.dcm);;All files (*.*)")            
        self.fileExtension = path.split(".")[-1] # get ext.
        
        # If no image chosen
        if path == "":
            return

        try:
            data = self.currentTab.setImage(path, self.fileExtension)
        except Exception as e:
            print(e)
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
            else:
                # If (jpeg, bitmap etc..)
                image = self.currentTab.primaryViewer.getOriginalImage()

                self.widthOfImage = self.getAttr(data,'width')
                self.heightOfImage = self.getAttr(data,'height')
                self.depthOfImage = getDepth(image)
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
        self.currentTab.clear()

    # Exit the application
    def exit(self):
        exitDialog = QMessageBox.critical(self,
        "Exit the application",
        "Are you sure you want to exit the application?",
        buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        defaultButton=QMessageBox.StandardButton.Yes)

        if exitDialog == QMessageBox.StandardButton.Yes.value:
            # Exit the application
            sys.exit()

    ##########################################
    #         """Image Functions"""          #
    ##########################################      

    # Transform to binary image
    def binaryImage(self):
        self.currentTab.primaryViewer.binaryImage()
    
    # Transform to negative image
    def negativeImage(self):
        self.currentTab.primaryViewer.negativeImage()

    # Log the image
    def logImage(self):
        self.currentTab.primaryViewer.logImage()

    # Gamma
    def gammaImage(self):
        requirements = {
            "Y":{
                "type": FLOAT,
                "range": (0, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Gamma Correction")
        if output != None:
            Y = output[0]
        else:
            return

        self.currentTab.primaryViewer.gammaCorrectionImage(Y)

    def contrastStretching(self):
        requirements = {
            "r1":{
                "type": INT,
                "range": (0, 255)
            },
            "s1":{
                "type": INT,
                "range": (0, 255)
            },
            "r2":{
                "type": INT,
                "range": (0, 255)
            },
            "s2":{
                "type": INT,
                "range": (0, 255)
            }
        }

        output = self.getInputsFromUser(requirements, "Contrast Stretching")
        if output != None:
            r1 = output[0]
            s1 = output[1]
            r2 = output[2]
            s2 = output[3]
        else:
            return

        self.currentTab.primaryViewer.contrastStretching(r1, s1, r2, s2)

    def intensityLevelSlicing(self):
        requirements = {
            "A":{
                "type": INT,
                "range": (0, 255)
            },
            "B":{
                "type": INT,
                "range": (0, 255)
            },
            "Mode":{
                "type": RADIO,
                "options": ["white","brightens"]
            }
        }

        output = self.getInputsFromUser(requirements, "Contrast Stretching")
        if output != None:
            A = output[0]
            B = output[1]
            mode = output[2]
        else:
            return

        if mode == "0":
            mode = "bw"
        else:
            mode = "bd"

        self.currentTab.primaryViewer.intensityLevelSlicing(A,B,mode)

    def bitPlaneSlicing(self):
        self.currentTab.primaryViewer.bitPlaneSlicing()

    # Get ROI            
    def getROI(self):
        image = self.currentTab.primaryViewer.getGrayImage()
        if len(image) != 0:
            _, ax = plt.subplots()
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            plt.title("Draw your region of interest (ROI)")
            plt.suptitle("Click Enter if you finished")
            
            def line_select_callback(eclick, erelease):
                ROI, mean, variance, std = self.currentTab.primaryViewer.setROI(toggle_selector.RS.corners)
                self.updateImage(ROI)
                self.setInfo("ROI", mean, variance, std)

            def handle_close(event):
                self.currentTab.primaryViewer.drawImage(image, scale="clip")
                self.updateImage(image)
                plt.close()

            def toggle_selector(event):
                if event.key == 'enter':
                    self.currentTab.primaryViewer.drawImage(image, scale="clip")
                    self.updateImage(image)
                    plt.close()

            toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                                drawtype='box', useblit=True,
                                                button=[1, 3],
                                                minspanx=3, minspany=3,
                                                spancoords='pixels',
                                                interactive=True)

            plt.connect('key_press_event', toggle_selector)
            plt.connect('close_event', handle_close)
            plt.show()

    ##########################################
    #         """Fourier Functions"""        #
    ##########################################


    ##########################################
    #       """Filters Functions"""          #
    ##########################################
    
    # Apply box filter
    def applyBoxFilter(self, mode="spatial"):
        requirements = {
            "Kernel size":{
                "type": INT,
                "range": (1, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Blur Image")
        if output != None:
            filterSize = output[0]
        else:
            return

        if filterSize > 0:
            if filterSize % 2 == 0:
                filterSize += 1

            if mode == "frequency":
                self.currentTab.primaryViewer.boxFilteringUsingFourier(filterSize)
            else:
                self.currentTab.primaryViewer.boxFiltering(filterSize)
                
    # Apply median masking
    def applyMedian(self): 
        requirements = {
            "Kernel size":{
                "type": INT,
                "range": (1, inf)
            },
            "Percent":{
                "type": FLOAT,
                "range": (0, 100)
            }
        }

        output = self.getInputsFromUser(requirements, "Order Statistic Filter")
        if output != None:
            filterSize = output[0]
            percent = output[1]
        else:
            return


        if filterSize > 0:
            if filterSize % 2 == 0:
                filterSize += 1
            
            self.currentTab.primaryViewer.OrderStatisticFilter(filterSize, percent)               
            
    # Apply un-sharp masking
    def applyUnsharp(self):

        requirements = {
            "Kernel size":{
                "type": INT,
                "range": (1, inf)
            },
            "HighBoost Factor":{
                "type": FLOAT,
                "range": (0, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Unsharp Filter")
        if output != None:
            filterSize = output[0]
            factorSize = output[1]
        else:
            return

        if filterSize % 2 == 0:
            filterSize += 1

        self.currentTab.primaryViewer.unsharpMask(filterSize,factorSize)
                
        # self.setInfo(self.interpolationMode, self.widthOfImage, self.heightOfImage, self.sizeOfImage, self.depthOfImage, self.modeOfImage, self.modalityOfImage, self.nameOfPatient, self.ageOfPatient, self.bodyOfPatient)
    
    # Apply notch reject filter
    def notchRejectFilter(self):
        image = self.currentTab.primaryViewer.getGrayImage()

        if len(image) != 0:
            requirements = {
                "Number of points":{
                    "type": INT,
                    "range": (1, inf)
                },
                "Radius":{
                    "type": FLOAT,
                    "range": (1, inf)
                }
            }

            output = self.getInputsFromUser(requirements, "Notch Reject Filter")
            if output != None:
                n = output[0]
                frequency = output[1]
            else:
                return

            plt.clf()
            spectrum = self.currentTab.primaryViewer.fourierTransform(image, draw=False)            
            magnitudeSpectrum = self.currentTab.primaryViewer.fourierTransform(image, mode="magnitude", log=True, draw=False)
            
            plt.imshow(magnitudeSpectrum, cmap = "gray")
            
            plt.title("Click on image to choose points. (Press any key to Start)", fontsize = 14)
            plt.waitforbuttonpress()
            plt.title(f'Select {n} points with mouse click', fontsize = 14)
            
            points = np.asarray(plt.ginput(n, timeout = -1))
            plt.close()

            self.currentTab.primaryViewer.notchRejectFilters(spectrum, points, frequency)
            
    ##########################################
    #     """Transformations Functions"""    #
    ##########################################

    # Shear Image
    def shearImage(self, mode="horizontal"):
        requirements = {
            "Shear Factor":{
                "type": FLOAT,
                "range": (-90, 90)
            }
        }

        output = self.getInputsFromUser(requirements, "Shear Image")
        if output != None:
            shearFactor = output[0]
        else:
            return

        self.currentTab.primaryViewer.shearImage(shearFactor, mode)
    
    # Zoom Image
    def zoomImage(self, mode="bilinear"):
        requirements = {
            "Zooming Factor":{
                "type": FLOAT,
                "range": (0.1, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Zoom image")
        if output != None:
            zoomingFactor = output[0]
        else:
            return
        
        self.widthOfImage, self.heightOfImage = self.currentTab.primaryViewer.zoomImage(zoomingFactor, mode)
                            
        if mode == "nearest":
            self.interpolationMode = "Zoom Nearest Neighbor"
        elif mode == "bilinear":
            self.interpolationMode = "Zoom Bilinear"

        self.setInfo(self.interpolationMode, self.widthOfImage, self.heightOfImage, self.sizeOfImage, self.depthOfImage, self.modeOfImage, self.modalityOfImage, self.nameOfPatient, self.ageOfPatient, self.bodyOfPatient)

    # Rotate Image
    def rotateImage(self, mode="bilinear"):
        requirements = {
            "Rotation Angle":{
                "type": FLOAT,
                "range": (-inf, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Rotate Image")
        if output != None:
            rotationAngle = output[0]
        else:
            return
        image = self.currentTab.primaryViewer.getGrayImage()
        self.widthOfImage, self.heightOfImage = self.currentTab.primaryViewer.rotateImage(image,rotationAngle, mode)

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
        self.statusbar.showMessage(f"Operation list has {len(self.images)} images",1000000)
        
    # get result of operation between two images
    def operationTwoImage(self, operation, images):
        if len(images) == 2:
            titleOfNewWindow = f"{operation}ing of images"
            newTab = self.addNewTab(titleOfNewWindow)            
            newTab.primaryViewer.operationTwoImages(operation, images[0], images[1])

    # get result of operation on one image
    def operationOneImage(self, operation):
        self.currentTab.primaryViewer.operationOneImages(operation)

    ##########################################
    #    """Construct Shapes Functions"""    #
    ##########################################

    # Draw Shapes
    def drawShape(self, shape="background"):
        if shape=="T":
            self.currentTab.primaryViewer.constructT()
        elif shape == "triangle":  
            self.currentTab.primaryViewer.constructTriangle()    
        elif shape == "circle":  
            self.currentTab.primaryViewer.constructCircle() 
        elif shape == "square":
            self.currentTab.primaryViewer.constructSquare()
        elif shape == "phantom":
            self.currentTab.primaryViewer.constructPhantom()
        else:
            self.currentTab.primaryViewer.constructBackground()

    ##########################################
    #         """Noise Functions"""          #
    ##########################################

    # Add random noise to the image
    def addNoise(self, mode="salt and pepper"):
        if mode == "uniform":
            requirements = {
                "a":{
                    "type": FLOAT,
                    "range": (-inf, inf)
                },
                "b":{
                    "type": FLOAT,
                    "range": (-inf, inf)
                }
            }

            output = self.getInputsFromUser(requirements, "Uniform Noise")
            if output != None:
                a = output[0]
                b = output[1]
            else:
                return
            
            self.currentTab.primaryViewer.addUniformNoise(a,b)
        elif mode == "gaussian":
            requirements = {
                "mean":{
                    "type": FLOAT,
                    "range": (-inf, inf)
                },
                "sigma":{
                    "type": FLOAT,
                    "range": (-inf, inf)
                }
            }

            output = self.getInputsFromUser(requirements, "Gaussian Noise")
            if output != None:
                mean = output[0]
                sigma = output[1]
            else:
                return

            self.currentTab.primaryViewer.addGaussianNoise(mean,sigma)
        elif mode == "rayleigh":
            requirements = {
                "mode":{
                    "type": FLOAT,
                    "range": (-inf, inf)
                }
            }

            output = self.getInputsFromUser(requirements, "Rayleigh Noise")
            if output != None:
                mode = output[0]
            else:
                return
            self.currentTab.primaryViewer.addRayleighNoise(mode)
        elif mode == "erlang":
            requirements = {
                "K":{
                    "type": FLOAT,
                    "range": (0, inf)
                },
                "Scale":{
                    "type": FLOAT,
                    "range": (0, inf)
                }
            }

            output = self.getInputsFromUser(requirements, "Erlang Noise")
            if output != None:
                k = output[0]
                scale = output[1]
            else:
                return
            self.currentTab.primaryViewer.addErlangNoise(k, scale)
        elif mode == "exponential":
            self.currentTab.primaryViewer.addExponentialNoise(5)
        elif mode == "pepper":
            self.currentTab.primaryViewer.addSaltAndPepperNoise("pepper")
        elif mode == "salt":
            self.currentTab.primaryViewer.addSaltAndPepperNoise("salt")
        elif mode == "salt and pepper":
            self.currentTab.primaryViewer.addSaltAndPepperNoise()

    ##########################################
    #  """Sinogram & Laminogram Functions""" #
    ##########################################

    def sinogram(self):
        # if self.currentTab.showHideSinogram():
        self.currentTab.sinogramViewer.drawSinogram(self.currentTab.primaryViewer.grayImage)
    
    def laminogram(self):
        # if self.currentTab.showHideLaminogram():
        requirements = {
            "Filter Type": {
                "type": RADIO,
                "options": ['None', 'Hamming', 'Ram-Lak (ramp)']
            },
            "Mode": {
                "type": RADIO,
                "options": ['Start, End, step',"Values"]
            },
            "Angles (comma ',') ":{
                "type": STR
            }
        }

        output = self.getInputsFromUser(requirements, "Laminogram")
        if output != None:
            filterType = output[0]
            mode = output[1]
            angles = output[2].replace(" ","")
        else:
            return
        angles = np.int64(np.array(angles.split(",")))

        if mode == 0:
            if len(angles) == 2:
                start, end = angles
                angles = range(start, end)
            elif len(angles) == 3:
                start, end, step = angles
                angles = range(start, end, step)

        if filterType == 0:
            self.currentTab.laminogramViewer.drawLaminogram(self.currentTab.sinogramViewer.grayImage, angles)
        elif filterType == 1:
            self.currentTab.laminogramViewer.drawLaminogram(self.currentTab.sinogramViewer.grayImage, angles, filterType="hamming")
        elif filterType == 2:
            self.currentTab.laminogramViewer.drawLaminogram(self.currentTab.sinogramViewer.grayImage, angles, filterType="ramp")

    ##########################################
    #         """View Functions"""           #
    ##########################################


    ##########################################

    # Base function  
    def baseBehavior(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
            self.updateImage()
        except Exception as e:
            print(e)
            QMessageBox.critical(self,"Error",f"Sorry, Error occurred. {e}")
            return  
                
    # Update magnitude and phase and histogram for the image after every update
    def updateImage(self, image=[], log=True):
        if image == []:
            image = self.currentTab.primaryViewer.grayImage
            
        if log:
            self.currentTab.magnitudeViewer.fourierTransform(image, "magnitude", True)
            self.currentTab.phaseViewer.fourierTransform(image, "phase", True)
        else:
            self.currentTab.magnitudeViewer.fourierTransform(image, "magnitude")
            self.currentTab.phaseViewer.fourierTransform(image, "phase")

        self.currentTab.histogramViewer.drawHistogram(image)
        # self.currentTab.sinogramViewer.drawSinogram(image)
                
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

    # Get inputs from user by give dictionary
    def getInputsFromUser(self, requirements, title="Blank"):
        inputWindow = popWindow(title, requirements)
        inputWindow.exec()
        
        result = list()
        output = inputWindow.getValues()
        loaded = inputWindow.checkLoaded()

        if loaded != False and output != None:
            for _, value in output.items():
                result.append(value)

            return tuple(result)

        return None

        