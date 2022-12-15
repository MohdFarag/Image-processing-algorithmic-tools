from math import *

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
        self.requirements = requirements.copy()
        self.inputs = requirements
        self.outputs = None
        self.loaded = False

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

    # Set values of inputs
    def setValues(self):
        self.outputs = self.inputs.copy()
        sentence = "NO ERROR"
        try:
            for key, value in self.requirements.items():  
                typeOfValue = value["type"]
                output = type(typeOfValue)(self.inputs[key].text())

                start, end = value["range"]
                if start != -inf and end != inf:
                    sentence = f"Sorry, The input must be between {start} and {end}."
                elif start == -inf and end != inf:
                    sentence = f"Sorry, The input must be less than {end}."
                elif start != -inf and end == inf:
                    sentence = f"Sorry, The input must be bigger than {start}."

                if start <= output <= end:
                    self.outputs[key] = output
                    self.loaded = True
                else:
                    QMessageBox.critical(self, "Error", sentence)
                    self.loaded = False
                    return 
        except Exception as e:
            print(e)
            self.loaded = False
            QMessageBox.critical(self, "Error", "Sorry, Error occurred.")
            return
        else:
            self.close()

    # Get values of inputs entered by user
    def getValues(self):
        return self.outputs

    # Check if process is done or canceled
    def checkLoaded(self):
        return self.loaded

    def cancelValues(self):
        self.outputs = None
        self.loaded = False
        self.close()

    def _connect(self):
        self.OkBtn.clicked.connect(self.setValues)
        self.CancelBtn.clicked.connect(self.cancelValues)