# Model-Maker

## About the Model Maker

With many model makers out for public use, there wasn't any that allowed for the user to make custom models easily. The main intentions of this model maker was to allow users to create custom image classifier models to use for their own use. With not many options for custom image classifiers I though the model maker would be a great idea to have for people that want their own custom models. The model maker application gives the users all the necessary inputs to create the model which once they are done entering those inputs can watch in the terminal to see the progress of creating the model. If the model is to their standards they can save the model or go back into it and add more photos, change inputs, etc.

## Usage

To start the model maker:
```Python
from PyQt5.QtWidgets import QApplication, QLineEdit, QWidget, QFormLayout, QPushButton, QComboBox, QTextEdit
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont
from PyQt5.QtCore import Qt
from Model_Creator_App import Model_Creator
import sys

app = QApplication(sys.argv)
win = Model_Creator()
win.show()
sys.exit(app.exec_())
```

Once you have launched the model maker application, the application lists some inputs needed in order to create the model:

1. The type of model (binary or categorical)
2. The horizontal size for when the application does image resizing
3. The vertical size for when the application does image resizing
4. The labels for the model
5. The epochs for when you train the model
6. The batch size for when you train the model

Once you have inserted all the necessary inputs, you have to create the model first. Then if the training went well and you have accurate results in the training the user is given the option to save the model to use in their own application.

## How to Install the Model Maker

In either cmd, powershell or Windows Terminal, type in:

```shell, sh, zsh, bash 
python Setup.py
```

Once you execute the python script, the script will prompt you with installing all the necessary libraries to run the Model Creator Application manually or to install the Model Creator Application. Once you have setup what you need to setup, simply type in **quit** to quit the application.

## License
[MIT](https://choosealicense.com/licenses/mit/)