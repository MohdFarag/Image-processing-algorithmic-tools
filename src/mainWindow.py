import numpy as np

# Resources
from .rcIcon import *

# Importing sys package
import sys

# Import Classes
from .tabViewer import tabViewer
from .popup import popWindow
from .utilities import *

# Style
import qdarktheme

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

        self.imageInformation = None
        self.SE = None
        
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
        self.unsharpMaskAction = QAction("&Un-sharp Mask", self)
        self.unsharpMaskAction.setStatusTip('Create a Unsharp Masking')

        ### Spatial filters ###
        
        # Spatial Box filter
        self.spatialBoxLpfAction = QAction("&Box", self)
        self.spatialBoxLpfAction.setStatusTip('Blur image by box kernel')

        # Spatial gaussian low pass filter
        self.spatialGaussianLpfAction = QAction("&Gaussian", self)
        self.spatialGaussianLpfAction.setStatusTip('Blur image by gaussian kernel')

        # Spatial Gradient filter
        self.spatialGradientHpfAction = QAction("&Gradient", self)
        self.spatialGradientHpfAction.setStatusTip('Apply Gradient filter on image')

        # Spatial Laplacian filter
        self.spatialLaplacianHpfAction = QAction("&Laplacian", self)
        self.spatialLaplacianHpfAction.setStatusTip('Apply Laplacian filter on image')

        ### Frequency filters ###
        
        ## Low pass filters ##
        # Ideal LPF
        self.frequencyIdealLpfAction = QAction("&Ideal", self)
        self.frequencyIdealLpfAction.setStatusTip('Apply ideal low pass filtering in frequency domain')

        # Frequency gaussian low pass filter
        self.frequencyGaussianLpfAction = QAction("&Gaussian", self)
        self.frequencyGaussianLpfAction.setStatusTip('Blur image by gaussian kernel')

        # Butterworth LPF
        self.frequencyButterworthLpfAction = QAction("&Butterworth", self)
        self.frequencyButterworthLpfAction.setStatusTip('Apply butterworth low pass filtering in frequency domain')

        ## High pass filters ##
        # Ideal HPF
        self.frequencyIdealHpfAction = QAction("&Ideal", self)
        self.frequencyIdealHpfAction.setStatusTip('Apply ideal high pass filtering in frequency domain')

        # Gaussian high pass filter
        self.frequencyGaussianHpfAction = QAction("&Gaussian", self)
        self.frequencyGaussianHpfAction.setStatusTip('sharpening image by gaussian kernel')

        # Butterworth HPF
        self.frequencyButterworthHpfAction = QAction("&Butterworth", self)
        self.frequencyButterworthHpfAction.setStatusTip('Apply butterworth high pass filtering in frequency domain')

        # Frequency Laplacian filter
        self.frequencyLaplacianHpfAction = QAction("&Laplacian", self)
        self.frequencyLaplacianHpfAction.setStatusTip('Apply Laplacian filter on image')

        ## Band filters ##
        # Notch reject filter
        self.notchRejectFilterAction = QAction("&Notch Reject", self)
        self.notchRejectFilterAction.setStatusTip('Apply notch reject filter on image')
        
        ## Order Statistic filter ##
        # Percentile filter [Min, Max, Median]
        self.percentilesFilterAction = QAction("&Percentiles", self)
        self.percentilesFilterAction.setStatusTip('Apply Percentiles filter on image')

        # Midpoint filter
        self.midpointFilterAction = QAction("&Midpoint", self)
        self.midpointFilterAction.setStatusTip('Apply midpoint filter on image')
        
        # Alpha trimmed mean filter
        self.alphaTrimmedMeanFilterAction = QAction("&Alpha Trimmed Mean", self)
        self.alphaTrimmedMeanFilterAction.setStatusTip('Apply alpha trimmed mean filter on image')

        # Homomorphic Filter Action
        self.homomorphicFilterAction = QAction("&Homomorphic", self)
        self.homomorphicFilterAction.setStatusTip('Apply Homomorphic Filter Action on image')

    # Transformation Actions
    def transformationsActions(self):

        # Scale nearest neighbour / Linear interpolation Action
        self.scaleAction = QAction(QIcon(":scale"), "&Scale", self)
        self.scaleAction.setStatusTip('Scale in/out by based on input')

        # Rotate the image by nearest neighbour / linear interpolation
        self.rotateAction = QAction(QIcon(":rotate"), "&Rotate", self)
        self.rotateAction.setStatusTip('Rotate the image')

        # Translate the image
        self.translateAction = QAction(QIcon(":translate"), "&Translate", self)
        self.translateAction.setStatusTip('Translate the image')
        
        # Shear the image
        self.shearAction = QAction(QIcon(":shear"), "&Shear", self)
        self.shearAction.setStatusTip('Shear the image')

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
        # Add uniform noise action
        self.addUniformNoiseAction = QAction(QIcon(":uniform"), "&Uniform Noise", self)
        self.addUniformNoiseAction.setStatusTip('Add uniform noise')

        self.addPeriodicNoiseAction = QAction(QIcon(":periodic"), "&Periodic Noise", self)
        self.addPeriodicNoiseAction.setStatusTip('Add periodic noise')
        
        # Add gaussian noise action
        self.addGaussianNoiseAction = QAction(QIcon(":gaussian"), "&Gaussian Noise", self)
        self.addGaussianNoiseAction.setStatusTip('Add gaussian noise')

        # Add rayleigh noise action
        self.addRayleighNoiseAction = QAction(QIcon(":Rayleigh"), "&Rayleigh Noise", self)
        self.addRayleighNoiseAction.setStatusTip('Add rayleigh noise')

        # Add erlang noise action
        self.addErlangNoiseAction = QAction(QIcon(":Erlang"), "&Erlang Noise", self)
        self.addErlangNoiseAction.setStatusTip('Add erlang noise')

        # Add exponential noise action
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
        
        "Fourier Actions"

        "Filters Actions"
        
        "Transformations Actions"

        "Operations Actions"
        self.addToCompareListAction.setShortcut("Ctrl++")

        "Construction Shapes Actions"

        "Noises Actions"

        "View Actions"

        "HelpActions"

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
        transformationMenu.addAction(self.scaleAction)
        transformationMenu.addAction(self.rotateAction)
        transformationMenu.addAction(self.translateAction)
        transformationMenu.addAction(self.shearAction)

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
        
        ### Spatial filters ###
        spatialMenu = QMenu("&Spatial Filters", self)
        ## Low pass filters ##
        spatialLpfMenu = QMenu("&Blurring Filters", self)
        spatialLpfMenu.addAction(self.spatialBoxLpfAction)
        spatialLpfMenu.addAction(self.spatialGaussianLpfAction)
        spatialMenu.addMenu(spatialLpfMenu)
        ## High pass filters ##
        spatialHpfMenu = QMenu("&Sharpening Filters", self)
        spatialHpfMenu.addAction(self.spatialGradientHpfAction)
        spatialHpfMenu.addAction(self.spatialLaplacianHpfAction)
        spatialMenu.addMenu(spatialHpfMenu)
        spatialMenu.addSeparator()
        ## Order statistics filters ##
        spatialOsfMenu = QMenu("&Order Statistics Filters", self)
        spatialOsfMenu.addAction(self.percentilesFilterAction)
        spatialOsfMenu.addAction(self.midpointFilterAction)
        spatialOsfMenu.addAction(self.alphaTrimmedMeanFilterAction)

        spatialMenu.addMenu(spatialOsfMenu)
        
        ### Frequency filters ###
        frequencyMenu = QMenu("&Frequency Filters", self)
        ## Low pass filters ##
        frequencyLpfMenu = QMenu("&Low Pass Filters", self)
        frequencyLpfMenu.addAction(self.frequencyIdealLpfAction)
        frequencyLpfMenu.addAction(self.frequencyGaussianLpfAction)
        frequencyLpfMenu.addAction(self.frequencyButterworthLpfAction)
        frequencyMenu.addMenu(frequencyLpfMenu)
        ## High pass filters ##
        frequencyHpfMenu = QMenu("&High Pass Filters", self)
        frequencyHpfMenu.addAction(self.frequencyIdealHpfAction)
        frequencyHpfMenu.addAction(self.frequencyGaussianHpfAction)
        frequencyHpfMenu.addAction(self.frequencyButterworthHpfAction)
        frequencyHpfMenu.addAction(self.frequencyLaplacianHpfAction)
        frequencyMenu.addMenu(frequencyHpfMenu)
        ## Notch pass filters ##
        frequencyMenu.addAction(self.notchRejectFilterAction)
        frequencyMenu.addSeparator()
        frequencyMenu.addAction(self.homomorphicFilterAction)
        
        filterMenu.addMenu(spatialMenu)
        filterMenu.addMenu(frequencyMenu)
        fileMenu.addSeparator()
        filterMenu.addAction(self.unsharpMaskAction)

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
        noiseMenu.addAction(self.addPeriodicNoiseAction)
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
        menuBar.addMenu(shapeMenu)
        menuBar.addMenu(filterMenu)
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
            
            self.toolBar.addAction(self.scaleAction)
            self.toolBar.addAction(self.rotateAction)
            self.toolBar.addAction(self.translateAction)     
            self.toolBar.addAction(self.shearAction)

        elif type == "shapes":
            self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.toolBar)  # type: ignore
            
            self.toolBar.addAction(self.constructTAction)
            self.toolBar.addAction(self.constructCircleBoxAction)
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

        self.dockInfo.setWidget(self.dataWidget)
        self.dockInfo.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dockInfo)

    # Set information
    def setInfo(self, image:np.ndarray, information):
        if image is not None:
            width =  image.shape[1]
            height = image.shape[0]
            
            depth = getDepth(image)
            size = width * height * depth

            modality = getAttribute(information, "Modality")
            patientName = getAttribute(information, "PatientName")
            patientAge = getAttribute(information,"PatientAge")
            bodyPartExamined = getAttribute(information,"BodyPartExamined") 

            info = {
                "Width": f"{width} px",
                "Height": f"{height} px",
                "Size": f"{size} bytes",
                "Bit depth": f"{depth} px",
                "Modality used": modality,
                "Patient name": patientName,
                "Patient Age": patientAge,
                "Body part examined": bodyPartExamined,
            }

            # Update the tree
            self.setDataOfTree(info)

    # Set the data of the tree
    def setDataOfTree(self, data):
        self.dataWidget.clear()
        i = 0
        for key, value in data.items():
            item = QTreeWidgetItem([key, str(value)])
            self.dataWidget.insertTopLevelItem(i, item)
            i += 1
    
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
        # Scale image
        self.scaleAction.triggered.connect(lambda: self.baseBehavior(self.scaleImage))
        # Rotate image
        self.rotateAction.triggered.connect(lambda: self.baseBehavior(self.rotateImage))
        # Translate image
        self.translateAction.triggered.connect(lambda: self.baseBehavior(self.translateImage))
        # Shear image
        self.shearAction.triggered.connect(lambda: self.baseBehavior(self.shearImage))

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
        ## Spatial filters ##
        # Low pass filters #
        self.spatialBoxLpfAction.triggered.connect(lambda: self.baseBehavior(self.applySpatialBoxLpf))
        self.spatialGaussianLpfAction.triggered.connect(lambda: self.baseBehavior(self.applySpatialGaussianLpf))
        # High pass filters #
        self.spatialGradientHpfAction.triggered.connect(lambda: self.baseBehavior(self.applySpatialGradientHpf)) 
        self.spatialLaplacianHpfAction.triggered.connect(lambda: self.baseBehavior(self.applySpatialLaplacianHpf))

        ## Frequency filters ##
        # Low pass filters #
        self.frequencyIdealLpfAction.triggered.connect(lambda: self.baseBehavior(self.applyFrequencyIdealLpf))
        self.frequencyGaussianLpfAction.triggered.connect(lambda: self.baseBehavior(self.applyFrequencyGaussianLpf))
        self.frequencyButterworthLpfAction.triggered.connect(lambda: self.baseBehavior(self.applyFrequencyButterworthLpf))
        # High pass filters #
        self.frequencyIdealHpfAction.triggered.connect(lambda: self.baseBehavior(self.applyFrequencyIdealHpf))
        self.frequencyGaussianHpfAction.triggered.connect(lambda: self.baseBehavior(self.applyFrequencyGaussianHpf))
        self.frequencyButterworthHpfAction.triggered.connect(lambda: self.baseBehavior(self.applyFrequencyButterworthHpf))
        self.frequencyLaplacianHpfAction.triggered.connect(lambda: self.baseBehavior(self.applyFrequencyLaplacianHpf))
        # Band pass filter #
        self.notchRejectFilterAction.triggered.connect(lambda: self.baseBehavior(self.applyNotchRejectFilter))

        self.unsharpMaskAction.triggered.connect(lambda: self.baseBehavior(self.applyUnsharp))
        self.percentilesFilterAction.triggered.connect(lambda: self.baseBehavior(self.applyPercentileFilter))
        self.midpointFilterAction.triggered.connect(lambda: self.baseBehavior(self.applyMidpointFilter))
        self.alphaTrimmedMeanFilterAction.triggered.connect(lambda: self.baseBehavior(self.applyAlphaTrimmedMeanFilter))
        self.homomorphicFilterAction.triggered.connect(lambda: self.baseBehavior(self.applyHomomorphicFilter))
        
        " Noise "
        self.addUniformNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "uniform"))
        self.addGaussianNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "gaussian"))
        self.addRayleighNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "rayleigh"))
        self.addErlangNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "erlang"))
        self.addExponentialNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "exponential"))
        self.addPeriodicNoiseAction.triggered.connect(lambda: self.baseBehavior(self.addNoise, "periodic"))
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
        fileExtension = path.split(".")[-1] # get ext.
        
        # If no image chosen
        if path == "":
            return

        try:
            self.imageInformation = self.currentTab.setImage(path, fileExtension)
        except Exception as e:
            print(e)
            appLogger.exception("Can't open the file !")
            QMessageBox.critical(self , "Corrupted image" , "Can't open the file !")
        else:
            self.statusbar.showMessage(path.split("/")[-1])
                
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
                "options": ["Black and White","Brightens or (Darkness)"]
            }
        }

        output = self.getInputsFromUser(requirements, "Contrast Stretching")
        if output != None:
            A = output[0]
            B = output[1]
            mode = output[2]
        else:
            return

        if mode == 0:
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
                ROI = self.currentTab.primaryViewer.setROI(toggle_selector.RS.corners)
                self.updateImage(ROI)

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
    
    ### Spatial Filters ###
    
    # Apply spatial box filter
    def applySpatialBoxLpf(self):
        requirements = {
            "Domain":{
                "type": RADIO,
                "options": DOMAINS
            },
            "Kernel size":{
                "type": INT,
                "range": (1, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Blur Image")
        if output != None:
            domain = DOMAINS[output[0]]
            filterSize = output[1]
        else:
            return

        if filterSize > 0:
            if filterSize % 2 == 0:
                filterSize += 1
               
            self.currentTab.primaryViewer.boxFiltering(filterSize, domain)

    # Apply spatial gaussian filter
    def applySpatialGaussianLpf(self):
        requirements = {
            "Kernel size":{
                "type": INT,
                "range": (1, inf)
            },
            "sigma":{
                "type": FLOAT,
                "range": (1, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Blur Image")
        if output != None:
            filterSize = output[0]
            sigma = output[1]
        else:
            return

        if filterSize % 2 == 0:
            filterSize += 1
            
        self.currentTab.primaryViewer.gaussianFiltering(sigma, filterSize)

    # Apply spatial gradient high pass filter
    # TODO:EDIT TO GRADIENT
    def applySpatialGradientHpf(self):    
        gradients = ["sobel","prewitt","roberts"] 
        SPECTRUMS = ["magnitude","phase","dx","dy"]
        
        requirements = {
            "Gradient Type":{
                "type": RADIO,
                "options": gradients
            },
            "Show":{
                "type": RADIO,
                "options": SPECTRUMS
            }
        }
        
        output = self.getInputsFromUser(requirements, "Gradient High Pass Filter")
        if output != None:
            gradientType = gradients[output[0]]
            show = SPECTRUMS[output[1]]
        else:
            return

        self.currentTab.primaryViewer.gradientFiltering(gradientType,show)

    # Apply spatial laplacian high pass filter
    def applySpatialLaplacianHpf(self):           
        self.currentTab.primaryViewer.laplacianFiltering("spatial")

    ### Frequency Filters ###
    
    # Apply frequency ideal low pass filter
    def applyFrequencyIdealLpf(self):
        requirements = {
            "Diameter":{
                "type": FLOAT,
                "range": (0.1, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Ideal Low Pass Filter")
        if output != None:
            diameter = output[0]
        else:
            return

        self.currentTab.primaryViewer.idealLowPassFilter(diameter)

    # Apply frequency gaussian low pass filter
    def applyFrequencyGaussianLpf(self):
        requirements = {
            "Diameter":{
                "type": FLOAT,
                "range": (0.1, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Gaussian Low Pass Filter")
        if output != None:
            diameter = output[0]
        else:
            return

        self.currentTab.primaryViewer.gaussianLowPassFilter(diameter)

    # Apply frequency butterworth low pass filter
    def applyFrequencyButterworthLpf(self):
        requirements = {
            "Diameter":{
                "type": FLOAT,
                "range": (0.1, inf)
            },
            "Order":{
                "type": FLOAT,
                "range": (-inf, inf)
            }

        }

        output = self.getInputsFromUser(requirements, "Butterworth Low Pass Filter")
        if output != None:
            diameter = output[0]
            n = output[1]
        else:
            return

        self.currentTab.primaryViewer.butterworthLowPassFilter(diameter, n)

    # Apply frequency ideal high pass filter
    def applyFrequencyIdealHpf(self):
        requirements = {
            "Diameter":{
                "type": FLOAT,
                "range": (0.1, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Ideal High Pass Filter")
        if output != None:
            diameter = output[0]
        else:
            return

        self.currentTab.primaryViewer.idealHighPassFilter(diameter)

    # Apply frequency gaussian high pass filter
    def applyFrequencyGaussianHpf(self):
        requirements = {
            "Diameter":{
                "type": FLOAT,
                "range": (0.1, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Gaussian High Pass Filter")
        if output != None:
            diameter = output[0]
        else:
            return

        self.currentTab.primaryViewer.gaussianHighPassFilter(diameter)

    # Apply frequency butterworth high pass filter
    def applyFrequencyButterworthHpf(self):
        requirements = {
            "Diameter":{
                "type": FLOAT,
                "range": (0.1, inf)
            },
            "Order":{
                "type": FLOAT,
                "range": (-inf, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Butterworth High Pass Filter")
        if output != None:
            diameter = output[0]
            n = output[1]
        else:
            return

        self.currentTab.primaryViewer.butterworthHighPassFilter(diameter, n)
    
    # Apply frequency laplacian high pass filter
    def applyFrequencyLaplacianHpf(self):           
        self.currentTab.primaryViewer.laplacianFiltering("frequency")

    ### Band Pass Filters ###
    
    # Apply notch reject filter
    def applyNotchRejectFilter(self):
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
            spectrum = fourierTransform(image)            
            magnitudeSpectrum = fourierTransform(image, mode="magnitude", log=True)
            
            plt.imshow(magnitudeSpectrum, cmap = "gray")
            
            plt.title("Click on image to choose points. (Press any key to Start)", fontsize = 14)
            plt.waitforbuttonpress()
            plt.title(f'Select {n} points with mouse click', fontsize = 14)
            
            points = np.asarray(plt.ginput(n, timeout = -1))
            plt.close()

            self.currentTab.primaryViewer.notchRejectFilter(spectrum, points, frequency)

    ### Order Statistic Filters ###
    
    # Apply order statistic filter masking
    def applyPercentileFilter(self): 
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
            
            self.currentTab.primaryViewer.percentileFilter(filterSize, percent)               

    # Apply midpoint filter masking
    def applyMidpointFilter(self): 
        requirements = {
            "Kernel size":{
                "type": INT,
                "range": (1, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Midpoint Filter")
        if output != None:
            filterSize = output[0]
        else:
            return


        if filterSize > 0:
            if filterSize % 2 == 0:
                filterSize += 1
            
            self.currentTab.primaryViewer.midPointFilter(filterSize)
    
    # Apply Alpha trimmed mean filter masking
    def applyAlphaTrimmedMeanFilter(self): 
        requirements = {
            "Kernel size":{
                "type": INT,
                "range": (1, inf)
            },
            "Alpha":{
                "type": INT,
                "range": (0, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Alpha Trimmed Mean Filter")
        if output != None:
            filterSize = output[0]
            alpha = output[1]
        else:
            return


        if filterSize % 2 == 0:
            filterSize += 1
        
        self.currentTab.primaryViewer.alphaTrimmedMeanFilter(filterSize, alpha)
    
    ### Other Filters ###
                
    # Apply un-sharp masking
    def applyUnsharp(self):
        requirements = {
            "Domain":{
                "type": RADIO,
                "options": DOMAINS,
            },
            "Filter Type":{
                "type": RADIO,
                "options": FILTER_TYPES,
            },
            "Kernel size/Diameter":{
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
            domain = DOMAINS[output[0]]
            filterType = FILTER_TYPES[output[1]]
            filterSize = output[2]
            factorSize = output[3]
        else:
            return

        if filterSize % 2 == 0 and domain == "Spatial":
            filterSize += 1

        self.currentTab.primaryViewer.unsharpMask(filterSize, factorSize, domain, filterType)
    
    # Apply Homomorphic Filter
    def applyHomomorphicFilter(self):
        requirements = {
            "Diameter":{
                "type": FLOAT,
                "range": (1, inf)
            },
            "Constant C":{
                "type": INT,
                "range": (1, inf)
            },
            "Gamma Low":{
                "type": FLOAT,
                "range": (0, inf)
            },
            "Gamma High":{
                "type": FLOAT,
                "range": (0, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Unsharp Filter")
        if output != None:
            d = output[0]
            c = output[1]
            gammaLow = output[2]
            gammaHigh = output[3]
        else:
            return

        self.currentTab.primaryViewer.homomorphicFilter(d, c, gammaLow, gammaHigh)
             
    ##########################################
    #     """Transformations Functions"""    #
    ##########################################

    # Shear Image
    def shearImage(self):
        requirements = {
            "Shear Type":{
                "type": RADIO,
                "options": ["horizontal","vertical"]
            },
            "Shear Factor":{
                "type": FLOAT,
                "range": (-90, 90)
            }
        }

        output = self.getInputsFromUser(requirements, "Shear Image")
        if output != None:
            shearDirection = output[0]
            shearFactor = output[1]
        else:
            return

        if shearDirection == 0:
            mode = "horizontal"
        else:
            mode = "vertical"
            
        self.currentTab.primaryViewer.shearImage(shearFactor, mode)
    
    # Scale Image
    def scaleImage(self):
        requirements = {
            "Interpolation Type":{
                "type": RADIO,
                "options": ["bilinear","nearest neighbor"]
            },
            "Scaling Factor":{
                "type": FLOAT,
                "range": (0.1, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Scale image")
        if output != None:
            interpolationType = output[0]
            scalingFactor = output[1]
        else:
            return
        
        if interpolationType == 0:
            mode = "bilinear"
        else:
            mode = "nearest neighbor"
            
        self.currentTab.primaryViewer.scaleImage(scalingFactor, mode)

    # Rotate Image
    def rotateImage(self):
        requirements = {
            "Interpolation Type":{
                "type": RADIO,
                "options": ["bilinear","nearest neighbor"]
            },
            "Rotation Angle":{
                "type": FLOAT,
                "range": (-inf, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Rotate Image")
        if output != None:
            interpolationType = output[0]
            rotationAngle = output[1]
        else:
            return
        
        if interpolationType == 0:
            mode = "bilinear"
        else:
            mode = "nearest neighbor"
            
        self.currentTab.primaryViewer.rotateImage(rotationAngle, mode)

    # Translate Image
    def translateImage(self):
        requirements = {
            "x":{
                "type": INT,
                "range": (-inf, inf)
            },
            "y":{
                "type": INT,
                "range": (-inf, inf)
            }
        }

        output = self.getInputsFromUser(requirements, "Translate Image")
        if output != None:
            x = output[0]
            y = output[1]
        else:
            return
            
        self.currentTab.primaryViewer.translateImage(x, y)

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
            requirements = {
                "Scale":{
                    "type": FLOAT,
                    "range": (0, inf)
                }
            }

            output = self.getInputsFromUser(requirements, "Exponential Noise")
            if output != None:
                scale = output[0]
            else:
                return
            self.currentTab.primaryViewer.addExponentialNoise(scale)            
        elif mode == "periodic":
            requirements = {
                "Amplitude":{
                    "type": FLOAT,
                    "range": (-inf, inf)
                },
                "Frequency":{
                    "type": FLOAT,
                    "range": (-inf, inf)
                },
                "Phase":{
                    "type": FLOAT,
                    "range": (-inf, inf)
                }
            }
            
            output = self.getInputsFromUser(requirements, "Periodic Noise")
            if output != None:
                amplitude = output[0]
                frequency = output[1]
                phase = output[2]
            else:
                return
            self.currentTab.primaryViewer.addPeriodicNoise(amplitude,frequency,phase)
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
        if self.currentTab.showHideSinogram():
            self.currentTab.sinogramViewer.drawSinogram(self.currentTab.primaryViewer.grayImage)
    
    def laminogram(self):
        if self.currentTab.showHideLaminogram():
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
            image = self.currentTab.primaryViewer.getGrayImage()

        if len(image) != 0:
            if log:
                self.currentTab.magnitudeViewer.fourierTransform(image, "magnitude", True)
                self.currentTab.phaseViewer.fourierTransform(image, "phase", True)
            else:
                self.currentTab.magnitudeViewer.fourierTransform(image, "magnitude", log=False)
                self.currentTab.phaseViewer.fourierTransform(image, "phase", log=False)

            self.currentTab.histogramViewer.drawHistogram(image)
            self.setInfo(image, self.imageInformation)
        
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

        