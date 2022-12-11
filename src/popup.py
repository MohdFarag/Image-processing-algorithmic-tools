# Importing Qt widgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# Importing Logging
from .log import appLogger

# Window class
class popWindow(QDialog):

    """Main Window"""
    def __init__(self, title:str, requirements:dict, *args, **kwargs):      
        super(popWindow, self).__init__(*args, **kwargs)

        """Variables"""
        self.inputs = requirements
        self.outputs = requirements

        """Setting Icon"""
        self.setWindowIcon(QIcon(":icon"))

        """Setting title"""
        self.setWindowTitle(title)

        """Central area"""
        self._initUI()

        """Connect signals"""
        self._connect()

    def _initUI(self):       
        # Outer Layout
        outerLayout = QVBoxLayout()

        for key, value in self.inputs.items():
            inputField = self.addInput(key)
            self.inputs[key] = inputField
            outerLayout.addWidget(inputField)

        self.CancelBtn = QPushButton("Cancel")
        self.OkBtn = QPushButton("Apply")

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(self.CancelBtn)
        buttonsLayout.addWidget(self.OkBtn)

        outerLayout.addLayout(buttonsLayout)
        self.setLayout(outerLayout)

    def addInput(self, placeholderText):
        inputField = QLineEdit()
        inputField.setPlaceholderText(placeholderText)
        inputField.setStyleSheet("""border:1px solid #00d; 
                                            height:18px; 
                                            padding:2px; 
                                            border-radius:5px; 
                                            font-size:16px; 
                                            margin-right:5px""")
        return inputField


    def setValues(self):
        for key, value in self.inputs.items():
            output = self.inputs[key].text()
            self.outputs[key] = output
        
        self.close()

    def getValues(self):
        return self.outputs

    def cancelValues(self):
        self.outputs = None
        self.close()

    def _connect(self):
        self.OkBtn.clicked.connect(self.setValues)
        self.CancelBtn.clicked.connect(self.cancelValues)