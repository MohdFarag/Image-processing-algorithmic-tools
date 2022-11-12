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

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class QVLine(QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)

class tabViewer(QWidget):
    def __init__(self, title:str="", color:str="red"):
        super(tabViewer, self).__init__()

        self.title = title
        self.color = color
        self.showHist = False
        
        # Initilize new layout
        self.tabLayout = QHBoxLayout()
        
        # Initilize the viewers
        self.primaryViewer = ImageViewer(type="image", title=self.title)
        self.tabLayout.addWidget(self.primaryViewer)

        self.histogramViewer = ImageViewer(axisExisting=True, axisColor=self.color, type="hist", title=f"Histogram of {self.title}")
        self.tabLayout.addWidget(self.histogramViewer)
        self.histogramViewer.hide()

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

    def setImage(self, path, fileExtension):
        data = self.primaryViewer.setImage(path, fileExtension)
        self.histogramViewer.drawHistogram(self.primaryViewer.grayImage)
        return data

    def drawT(self):
        self.primaryViewer.constructT("white")
        self.histogramViewer.drawHistogram(self.primaryViewer.grayImage)
