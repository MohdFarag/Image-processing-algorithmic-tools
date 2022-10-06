# importing Qt widgets
from PyQt5.QtCore import QObject, pyqtSignal

class MapWorker(QObject):
    
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    
    def __init__(self):
        super(MapWorker, self).__init__()
        self._isRunning = True

    def run(self, timePlot, mainDataPlot, xErrorMap, yErrorMap): 
        if not self._isRunning :
            self._isRunning = True

        self.progress.emit(int(self.progress+1))

        self.finished.emit()


    def stop(self):
        self._isRunning = False