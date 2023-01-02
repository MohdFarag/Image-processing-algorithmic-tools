# Importing Qt widgets
try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtGui import *
    from PyQt6.QtCore import *
except ImportError:
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *

from .image import ImageViewer
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
    
class tabViewer(QWidget):
    """Tab widget with 7 viewers"""
    primaryViewer: ImageViewer
    secondaryViewer: ImageViewer
    histogramViewer: ImageViewer
    magnitudeViewer: ImageViewer
    phaseViewer: ImageViewer
    sinogramViewer: ImageViewer
    laminogramViewer: ImageViewer
    
    def __init__(self, title:str="", color:str="red", type="normal"):
        super(tabViewer, self).__init__()

        self.title = title
        self.color = color
        self.showHist = False
        self.showMag = False
        self.showPhase = False
        self.showSinogram = False
        self.showLaminogram = False
        
        # Initialize new layout
        self.tabLayout = QGridLayout()
        
        # Initialize the viewers
        self.primaryViewer, self.primaryFrame = self.addViewer(type="image", title=self.title)
        
        self.secondaryViewer, self.secondaryFrame = self.addViewer(type="image", title=self.title)

        # Histogram viewers
        self.histogramViewer, self.histogramFrame = self.addViewer(axisExisting=True, axisColor=self.color, type="hist", title=f"Histogram of {self.title}")
        
        # Frequencies components
        self.magnitudeViewer, self.magnitudeFrame = self.addViewer(axisExisting=True, axisColor=self.color, title=f"Magnitude of {self.title}")
        self.phaseViewer, self.phaseFrame = self.addViewer(axisExisting=True, axisColor=self.color, title=f"Phase of {self.title}")

        # Phantom components
        self.sinogramViewer, self.sinogramFrame = self.addViewer(axisExisting=True, axisColor=self.color, title=f"Sinogram of {self.title}")
        self.laminogramViewer, self.laminogramFrame = self.addViewer(axisExisting=True, axisColor=self.color, title=f"Laminogram of {self.title}")

        self.tabLayout.addWidget(self.primaryFrame,0,0)
        if type == "compare":
            self.tabLayout.addWidget(self.secondaryFrame,0,1)
        else:
            self.tabLayout.addWidget(self.histogramFrame,0,1)
            self.histogramFrame.hide()
            
            self.tabLayout.addWidget(self.magnitudeFrame,1,0)
            self.magnitudeFrame.hide()

            self.tabLayout.addWidget(self.phaseFrame,1,1)
            self.phaseFrame.hide()

            self.tabLayout.addWidget(self.sinogramFrame,2,0)
            self.sinogramFrame.hide()

            self.tabLayout.addWidget(self.laminogramFrame,2,1)
            self.laminogramFrame.hide()

        # Set layout to new tab
        self.setLayout(self.tabLayout)

    def addViewer(self,axisExisting=False, axisColor="#329da8", type="image", title=""):
        viewerFrame = QFrame()
        viewerLayout = QVBoxLayout()
        viewer = ImageViewer(None,axisExisting, axisColor, type, title)
        navToolbar = NavigationToolbar(viewer)

        viewerLayout.addWidget(navToolbar)
        viewerLayout.addWidget(viewer)
        viewerFrame.setLayout(viewerLayout)

        return viewer, viewerFrame

    def showHideHistogram(self):
        if not self.showHist:
            self.showHist = True
            self.histogramFrame.show()
        else:
            self.showHist = False
            self.histogramFrame.hide()
    
    def showHideMagnitude(self):
        if not self.showMag:
            self.showMag = True
            self.magnitudeFrame.show()
        else:
            self.showMag = False
            self.magnitudeFrame.hide()
    
    def showHidePhase(self):
        if not self.showPhase:
            self.showPhase = True
            self.phaseFrame.show()
        else:
            self.showPhase = False
            self.phaseFrame.hide()

    def showHideFourier(self):
        self.showHideMagnitude()
        self.showHidePhase()

    def showHideSinogram(self):
        if not self.showSinogram:
            self.showSinogram = True
            self.sinogramFrame.show()
        else:
            self.showSinogram = False
            self.sinogramFrame.hide()
        
        return self.showSinogram
    
    def showHideLaminogram(self):
        if not self.showLaminogram:
            self.showLaminogram = True
            self.laminogramFrame.show()
        else:
            self.showLaminogram = False
            self.laminogramFrame.hide()
        
        return self.showLaminogram

    def equalize(self):
        self.primaryViewer.drawNormalizedHistogram()

    def setImage(self, path, fileExtension):
        data = self.primaryViewer.setImage(path, fileExtension)
        self.magnitudeViewer.fourierTransform(self.primaryViewer.grayImage,"magnitude")
        self.phaseViewer.fourierTransform(self.primaryViewer.grayImage,"phase")

        return data

    def clear(self):
        self.primaryViewer.reset()
        self.histogramViewer.reset()
        self.magnitudeViewer.reset()
        self.phaseViewer.reset()
        self.sinogramViewer.reset()
        self.laminogramViewer.reset()