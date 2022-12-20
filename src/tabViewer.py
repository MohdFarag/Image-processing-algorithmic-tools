# importing Qt widgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from .image import ImageViewer

class tabViewer(QWidget):
    """Tab widget with 4 viewers"""
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
        self.primaryViewer = ImageViewer(type="image", title=self.title)
        self.secondaryViewer = ImageViewer(type="image", title=self.title)

        # Histogram viewers
        self.histogramViewer = ImageViewer(axisExisting=True, axisColor=self.color, type="hist", title=f"Histogram of {self.title}")
        
        # Frequencies components
        self.magnitudeViewer = ImageViewer(axisExisting=True, axisColor=self.color, title=f"Magnitude of {self.title}")
        self.phaseViewer = ImageViewer(axisExisting=True, axisColor=self.color, title=f"Phase of {self.title}")

        # Phantom components
        self.sinogramViewer = ImageViewer(axisExisting=True, axisColor=self.color, title=f"Sinogram of {self.title}")
        self.laminogramViewer = ImageViewer(axisExisting=True, axisColor=self.color, title=f"Laminogram of {self.title}")

        self.tabLayout.addWidget(self.primaryViewer,0,0)
        if type == "compare":
            self.tabLayout.addWidget(self.secondaryViewer,0,1)
        else:
            self.tabLayout.addWidget(self.histogramViewer,0,1)
            self.histogramViewer.hide()
            
            self.tabLayout.addWidget(self.magnitudeViewer,1,0)
            self.magnitudeViewer.hide()

            self.tabLayout.addWidget(self.phaseViewer,1,1)
            self.phaseViewer.hide()

            self.tabLayout.addWidget(self.sinogramViewer,2,0)
            self.sinogramViewer.hide()

            self.tabLayout.addWidget(self.laminogramViewer,2,1)
            self.laminogramViewer.hide()

        # Set layout to new tab
        self.setLayout(self.tabLayout)

    def showHideHistogram(self):
        if not self.showHist:
            self.showHist = True
            self.histogramViewer.show()
        else:
            self.showHist = False
            self.histogramViewer.hide()
    
    def showHideMagnitude(self):
        if not self.showMag:
            self.showMag = True
            self.magnitudeViewer.show()
        else:
            self.showMag = False
            self.magnitudeViewer.hide()
    
    def showHidePhase(self):
        if not self.showPhase:
            self.showPhase = True
            self.phaseViewer.show()
        else:
            self.showPhase = False
            self.phaseViewer.hide()

    def showHideFourier(self):
        self.showHideMagnitude()
        self.showHidePhase()

    def showHideSinogram(self):
        if not self.showSinogram:
            self.showSinogram = True
            self.sinogramViewer.show()
        else:
            self.showSinogram = False
            self.sinogramViewer.hide()
        
        return self.showSinogram
    
    def showHideLaminogram(self):
        if not self.showLaminogram:
            self.showLaminogram = True
            self.laminogramViewer.show()
        else:
            self.showLaminogram = False
            self.laminogramViewer.hide()
        
        return self.showLaminogram

    def equalize(self):
        self.primaryViewer.drawNormalizedHistogram()

    def setImage(self, path, fileExtension):
        data = self.primaryViewer.setImage(path, fileExtension)
        self.magnitudeViewer.fourierTransform(self.primaryViewer.grayImage,"magnitude")
        self.phaseViewer.fourierTransform(self.primaryViewer.grayImage,"phase")

        return data