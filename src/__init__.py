# !/usr/bin/python

import sys
from app import *
from style import progressBarStyle


if __name__ == "__main__":

    # Create the application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(progressBarStyle)

    win = Window()
    win.show()

    # Create and show the main window
    win = Window()
    win.show()
        
    # Run the event loop
    sys.exit(app.exec_())