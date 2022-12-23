from math import *

# Importing Qt widgets
try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtGui import *
    from PyQt6.QtCore import *
except ImportError:
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *

from .utilities import *

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
        self.outerLayout = QVBoxLayout()

        for key, value in self.inputs.items():
            inputField = self.addInput(key, value.get("type"), value.get("options"))
            self.inputs[key] = inputField

        self.CancelBtn = QPushButton("Cancel")
        self.OkBtn = QPushButton("Apply")

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(self.CancelBtn)
        buttonsLayout.addWidget(self.OkBtn)

        self.outerLayout.addLayout(buttonsLayout)
        self.setLayout(self.outerLayout)

    def addInput(self, placeholderText, type="", items=None):
        if type == RADIO:
            radioLayout = QHBoxLayout()
            radioGroup = QButtonGroup(self)
            radioList = list()
            for item in items: 
                radioBtn = QRadioButton(item, self)
                radioGroup.addButton(radioBtn)
                radioLayout.addWidget(radioBtn)
                radioList.append(radioBtn)
            self.outerLayout.addLayout(radioLayout)
            return radioList
        else:
            inputField = QLineEdit()
            inputField.setPlaceholderText(placeholderText)
            inputField.setStyleSheet("""border:1px solid #00d; 
                                                height:18px; 
                                                padding:2px; 
                                                border-radius:5px; 
                                                font-size:16px; 
                                                margin-right:5px""")
            self.outerLayout.addWidget(inputField)
            return inputField

    # Set values of inputs
    def setValues(self):
        self.outputs = self.inputs.copy()
        sentence = "NO ERROR"
        state = True
        
        try:
            for key, value in self.requirements.items():  
                typeOfValue = value.get("type")
                if typeOfValue != RADIO:
                    output = type(typeOfValue)(self.inputs[key].text())
                else:
                    output = -1
                    for radioBtn in self.inputs[key]:
                        output += 1
                        if radioBtn.isChecked():
                            break
                                       
                if output != "" and output != -1:
                    if value.get("range") != None:
                        start, end = value.get("range")
                        if start != -inf and end != inf:
                            sentence = f"Sorry, The input must be between {start} and {end}."
                        elif start == -inf and end != inf:
                            sentence = f"Sorry, The input must be less than {end}."
                        elif start != -inf and end == inf:
                            sentence = f"Sorry, The input must be bigger than {start}."
                        state = start <= output <= end
                else:
                    sentence = "Please, Enter empty values."
                    state = False
                
                if state:
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