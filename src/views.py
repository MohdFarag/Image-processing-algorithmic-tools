# Resources
from .rcIcon import *

# Importing sys package
import sys
import os

# Import Classes
from .additionsQt import *

# Importing Qt widgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Importing numpy and pandas
import numpy as np
import pandas as pd

# Importing pyqtgraph as pg
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from pyqtgraph.dockarea import *

# Importing Logging
from .log import appLogger

class Window(QMainWindow):
    """Main Window."""
    def __init__(self):
        """Initializer."""
        super().__init__()

        ### Initialize Variable
        self.thread = QThread()

        ### Setting Icon
        self.setWindowIcon(QIcon(":icon.svg"))

        ### Setting title
        self.setWindowTitle("Image Viewer")

        ### UI contents
        self._createActions()
        self._createMenuBar()
        self._createToolBar()
        # self._createContextMenu()
        self._createStatusBar()
        self._connectActions()
        # Central area
        self._initUI()
        # Connect signals
        self._connect()
    
    # Set theme
    def setTheme(self, theme):
        pass
    
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

        appLogger.debug("Application started successfully.")

    # Tool Bar
    def _createToolBar(self):
        # Using a title
        toolBar = self.addToolBar("File")
        toolBar.addAction(self.openAction)
    
    # Context Menu
    def _createContextMenu(self):
        # Setting contextMenuPolicy
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        # Populating the widget with actions
        self.addAction(self.openAction)
        
        self.addSeperator(self)

        self.addAction(self.helpContentAction)
        self.addAction(self.checkUpdatesAction)
        self.addAction(self.aboutAction)

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
        self.size = QLabel(f"{self.getSize()}")
        self.statusbar.addPermanentWidget(self.size)

    # GUI
    def _initUI(self):
        centralMainWindow = QWidget(self)
        self.setCentralWidget(centralMainWindow)
        # Outer Layout
        outerLayout = QVBoxLayout()

        ######### INIT GUI #########
        # Main layout

        ######### INIT GUI #########
        centralMainWindow.setLayout(outerLayout)

    def getSize(self):
        return 0 

    # Connect
    def _connectActions(self):
        self.openAction.triggered.connect(self.browseImage)
        self.exitAction.triggered.connect(self.exit)
    
    def _connect(self):
        pass

    # Open image
    def browseImage(self):
        try:
            path, fileExtension = QFileDialog.getOpenFileName(None, "Load Image File", os.getenv('HOME') ,"csv(*.csv)")            
            # If no message chosed
            if path == "":
                return
            
            if fileExtension == "csv(*.csv)":
                pass

        except Exception as e:
            appLogger.exception("Can't open a image file")
    
    def addSeperator(self, parent):
        # Creating a separator action
        self.separator = QAction(self)
        self.separator.setSeparator(True)
        parent.addAction(self.separator)

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