import sys

from PyQt5.QtWidgets import QApplication
from .views import Window

from .style import *

# Importing Logging
from .log import appLogger

def main():
    # Create the application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create and show the main window
    win = Window()
    
    # Set stylesheet
    qtmodern.styles.light(app)
    win = qtmodern.windows.ModernWindow(win)
    
    win.show()
    appLogger.debug("Application started successfully.")

    # Run the event loop
    sys.exit(app.exec_())