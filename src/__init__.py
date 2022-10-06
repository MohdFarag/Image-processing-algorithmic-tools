# !/usr/bin/python

import sys
from app import *
from style import progressBarStyle


if __name__ == "__main__":

    # Initialize Our Window App
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(progressBarStyle)

    popWindow = Window()
    popWindow.show()

    # Run the application
    sys.exit(app.exec_())