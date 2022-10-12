# Resources
import PIL
import pydicom
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

        self.infoDict = dict()

        ### Setting Icon
        self.setWindowIcon(QIcon(":icon.svg"))

        ### Setting title
        self.setWindowTitle("Image Viewer")

        ### UI contents
        self._createActions()
        self._createMenuBar()
        self._createToolBar()
        self._createStatusBar()
        # Central area
        self._initUI()
        # Connect signals
        self._connect()
    
    # Actions
    def _createActions(self):
        # Open Action
        self.openAction = QAction(QIcon(":file.ico"), "&Open...", self)
        self.openAction.setShortcut("Ctrl+O")
        self.openAction.setStatusTip('Open a new image')

        # Exit Action
        self.exitAction = QAction(QIcon(":exit.svg"), "&Exit...", self)
        self.exitAction.setShortcut("Ctrl+Q")
        self.exitAction.setStatusTip('Exit application')

        # Help Actions
        self.helpContentAction = QAction("&Help Content", self)
        self.helpContentAction.setStatusTip('Help')
        self.checkUpdatesAction = QAction("&Check For Updates", self)
        self.checkUpdatesAction.setStatusTip('Check Updates')
        self.aboutAction = QAction("&About...", self)
        self.aboutAction.setStatusTip('About')

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
        fileMenu.addAction(self.exitAction) # Exit file in menu
        
        ## Help tap
        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addSeparator() # Seperator
        helpMenu.addAction(self.checkUpdatesAction)
        helpMenu.addSeparator() # Seperator
        helpMenu.addAction(self.aboutAction)
        
        ## Append taps
        menuBar.addMenu(fileMenu)
        menuBar.addMenu(helpMenu)

    # Tool Bar
    def _createToolBar(self):
        # Using a title
        toolBar = self.addToolBar("File")
        toolBar.addAction(self.openAction)
    
    # Context Menu Event
    def contextMenuEvent(self, event):
        # Creating a menu object with the central widget as parent
        menu = QMenu(self)
        # Populating the menu with actions
        menu.addAction(self.openAction)
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

        ### init GUI ###
        
        # Main layout
        
        self.viewer = ImageViewer()
        outerLayout.addWidget(self.viewer)

        # Add docker
        self.addDockLayout()

        ### GUI ###
        centralMainWindow.setLayout(outerLayout)


    def addDockLayout(self):
        # Dock widget    
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
        self.exitAction.triggered.connect(self.exit) # When click on exit action
    
    def _connect(self):
        self._connectActions()

    # Open image
    def browseImage(self):
            # Browse function
            path, _ = QFileDialog.getOpenFileName(None, "Load Image File", filter="Custom files (*.bmp *.jpeg *.jpg *.dcm);;All files (*.*)")            
            fileExtension = path.split(".")[-1] # get ext.
            
            # If no image choosed
            if path == "":
                return
            
            try:
                data = self.viewer.setImage(path,fileExtension)
            except:
                # Error
                appLogger.exception("Can't open the file !")
                QMessageBox.critical(self , "Corrupted image" , "Sorry, the image is corrupted !")
            else:
                self.statusbar.showMessage(path.split("/")[-1])

                # To convert to bits: multiply by 8
                size = f"{os.stat(path).st_size * 8} bits" 

                if fileExtension == "dcm":
                    # If dicom
                    width =  f"{self.getAttr(data, 'Rows')} px"
                    height = f"{self.getAttr(data, 'Columns')} px"
                    depth = f"{self.getAttr(data, 'BitsStored')} bit/pixel"
                    mode = self.getAttr(data, "PhotometricInterpretation")
                    modality = self.getAttr(data, "Modality")
                    name = self.getAttr(data, "PatientName")
                    age = self.getAttr(data,"PatientAge")
                    body = self.getAttr(data,"BodyPartExamined") 
                    # Set the information                  
                    self.setInfo(fileExtension,width, height, size, depth, mode, modality, name, age, body)
                else:
                    # If (jpeg, bitmap)
                    width = f"{self.getAttr(data,'width')} px"
                    height = f"{self.getAttr(data,'height')} px"
                    depth = f"{MODE_TO_BPP[data.mode]} bit/pixel"
                    mode = self.getAttr(data,"mode")
                    # Set the information
                    self.setInfo(fileExtension, width, height, size, depth, mode)
    
    # Get Data
    def getAttr(self, variable, att):
        if hasattr(variable, att):
            # If attribute is found.
            return getattr(variable, att)
        else:
            # If attribute is not found.
            return "N/A"

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