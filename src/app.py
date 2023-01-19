"""Main application file for the application."""

import sys

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:
    from PyQt5.QtWidgets import QApplication

from .mainWindow import MainWindow, qdarktheme

# Importing Logging
from .log import appLogger

def main():
    """Main function for the application."""

    # Create the application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Apply the complete light theme to Qt App.
    qdarktheme.setup_theme("light")

    # Create and show the main window
    window = MainWindow()
    window.show()

    appLogger.debug("Application started successfully.")
    print("Application started successfully.")

    # Start the event loop
    sys.exit(app.exec())
