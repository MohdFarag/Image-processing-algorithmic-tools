# importing Qt widgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from .image import ImageViewer

class TableView(QTableWidget):
    def __init__(self,*args):
        QTableWidget.__init__(self, *args)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(['Frequency','Magnitude','Phase'])
 
    def addData(self, frequency, magnitude, phase): 
        rowPosition = self.rowCount()

        self.insertRow(rowPosition)
        self.setItem(rowPosition , 0, QTableWidgetItem(f"{frequency}Hz"))
        self.setItem(rowPosition , 1, QTableWidgetItem(f"{magnitude}"))
        self.setItem(rowPosition , 2, QTableWidgetItem(f"{phase}°"))

    def clearAllData(self):
        self.setRowCount(0)

class tabViewer(QWidget):
    def __init__(self, title:str="", color:str="red"):
        super(tabViewer, self).__init__()

        self.title = title
        self.color = color
        self.showHist = False
        self.showMag = False
        self.showPhase = False
        
        # Initilize new layout
        self.tabLayout = QGridLayout()
        
        # Initilize the viewers
        self.primaryViewer = ImageViewer(type="image", title=self.title)
        self.tabLayout.addWidget(self.primaryViewer,0,0)

        self.histogramViewer = ImageViewer(axisExisting=True, axisColor=self.color, type="hist", title=f"Histogram of {self.title}")
        self.tabLayout.addWidget(self.histogramViewer,0,1)
        self.histogramViewer.hide()

        self.magnitudeViewer = ImageViewer(axisExisting=True, axisColor=self.color, title=f"Magnitude of {self.title}")
        self.tabLayout.addWidget(self.magnitudeViewer,1,0)
        self.magnitudeViewer.hide()

        self.phaseViewer = ImageViewer(axisExisting=True, axisColor=self.color, title=f"Phase of {self.title}")
        self.tabLayout.addWidget(self.phaseViewer,1,1)
        self.phaseViewer.hide()

        # Set layout to new tab
        self.setLayout(self.tabLayout)

    def equalize(self):
            self.primaryViewer.normalizeHistogram()
            self.histogramViewer.drawHistogram(self.primaryViewer.grayImage)

    def addHistogram(self):
        if not self.showHist:
            self.showHist = True
            self.histogramViewer.show()
        else:
            self.showHist = False
            self.histogramViewer.hide()
    
    def addMagnitude(self):
        if not self.showMag:
            self.showMag = True
            self.magnitudeViewer.show()
        else:
            self.showMag = False
            self.magnitudeViewer.hide()
    
    def addPhase(self):
        if not self.showPhase:
            self.showPhase = True
            self.phaseViewer.show()
        else:
            self.showPhase = False
            self.phaseViewer.hide()

    def setImage(self, path, fileExtension):
        data = self.primaryViewer.setImage(path, fileExtension)
        self.histogramViewer.drawHistogram(self.primaryViewer.grayImage)
        self.magnitudeViewer.fourierTransform(self.primaryViewer.grayImage,"magnitude")
        self.phaseViewer.fourierTransform(self.primaryViewer.grayImage,"phase")

        return data

    def drawT(self):
        self.primaryViewer.constructT("white")
        self.histogramViewer.drawHistogram(self.primaryViewer.grayImage)

    def drawTriangle(self):
        self.primaryViewer.constructTriangle("white")
        self.histogramViewer.drawHistogram(self.primaryViewer.grayImage)
