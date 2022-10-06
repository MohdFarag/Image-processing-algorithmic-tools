# Definition of Main Color Palette
from Defs import COLOR1,COLOR2,COLOR3,COLOR4, COLOR5

# Importing sys package
import sys
import os

# Importing threads
from Threads import *

# Import Classes
from additionsQt import *

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
from log import app_logger

class Window(QMainWindow):
    """Main Window."""
    def __init__(self):
        """Initializer."""
        super().__init__()

        ### Initialize Variable
        self.thread = QThread()

        ### Setting Icon
        self.setWindowIcon(QIcon('images/icon.ico'))

        ### Setting title
        self.setWindowTitle("Image Viewer")

        ### UI contents
        self.createMenuBar()
        self.initUI()
        ### Status Bar
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet(f"""font-size:13px;
                                 padding: 3px;
                                 color: {COLOR1};
                                 font-weight:900;""")
        self.statusBar.showMessage("Welcome to our application...")
        self.setStatusBar(self.statusBar)
        ### Connect action
        self.connect()
    
    # Set theme
    def setTheme(self, theme):
        pass

    # Menu
    def createMenuBar(self):
        # MenuBar
        menuBar = self.menuBar()

        # Creating menus using a QMenu object
        fileMenu = QMenu("&File", self)

        # Open file in menu
        self.openFile = QAction("Open...",self)
        self.openFile.setShortcut("Ctrl+o")
        self.openFile.setStatusTip('Open a new image')
        self.openFile.triggered.connect(self.browseSignal)

        fileMenu.addAction(self.openFile)

        # Exit file in menu
        self.quit = QAction("Exit",self)
        self.quit.setShortcut("Ctrl+q")
        self.quit.setStatusTip('Exit application')

        fileMenu.addAction(self.quit)

        # Add file tab to the menu
        menuBar.addMenu(fileMenu)

        app_logger.debug("Application started successfully.")

    # GUI
    def initUI(self):
        centralMainWindow = QWidget(self)
        self.setCentralWidget(centralMainWindow)
        # Outer Layout
        outerLayout = QVBoxLayout()

        ######### INIT GUI #########
        # Main layout

        ######### INIT GUI #########
        centralMainWindow.setLayout(outerLayout)

    def connect(self):
        pass
    
    # Open image
    def browseImage(self):
        try:
            path, fileExtension = QFileDialog.getOpenFileName(None, "Load Signal File", os.getenv('HOME') ,"csv(*.csv)")            
            # If no message chosed
            if path == "":
                return
            
            if fileExtension == "csv(*.csv)":
                pass

        except Exception as e:
            app_logger.exception("Can't open a csv file")
            
    def reportProgress(self, n):
        self.progressbar.setValue(n)
    
    # Exit the application
    def exit(self):
        exitDlg = QMessageBox.critical(self,
        "Exit the application",
        "Are you sure you want to exit the application?",
        buttons=QMessageBox.Yes | QMessageBox.No,
        defaultButton=QMessageBox.No)

        if exitDlg == QMessageBox.Yes:
            # Exit the application
            sys.exit()