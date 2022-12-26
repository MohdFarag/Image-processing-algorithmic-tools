import sys

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:
    from PyQt5.QtWidgets import QApplication

from .mainWindow import *

# Importing Logging
from .log import appLogger

def main():
    # Create the application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Apply the complete light theme to Qt App.
    qdarktheme.setup_theme("auto")
    
    # Create and show the main window
    window = MainWindow()
    window.show()

    appLogger.debug("Application started successfully.")

    # Start the event loop
    sys.exit(app.exec())