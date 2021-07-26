# GUI Libraries
from PyQt5.QtWidgets import QApplication, QLineEdit, QWidget, QFormLayout, QPushButton, QComboBox, QTextEdit
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont
from PyQt5.QtCore import Qt
import sys, pickle

# Multi Image Classifier Library
from Multi_Classification.Multi_Image_Classification import Multi_Image_Classification as img_classifier

# Binary Image Classifier Library
from Binary_Classification.Image_Classification import Image_Classification as bin_classifier

'''
Model Creator Class

Description: 
1.  An application to make it easy for a user to create a model for their dataset.
2.  Uses categorical classification and binary classification depending on the user's choice.
3.  Currently uses PyQt5 as the base for the GUI.
'''
class Model_Creator(QWidget):

    h_size_input = None # make a variable to contain the horizontal size
    v_size_input = None # make a variable to contain the vertical size
    labels_input = None # make a variable to contain the labels input
    model_type_input = None # make a variable to contain the type of model (binary or categorical)

    classifier = None # contain the model created (either binary or categorical)
    
    # create a general textbox with a int constraint
    def create_textbox_int(self, max_length):
        textbox = QLineEdit()
        textbox.setValidator(QIntValidator())
        textbox.setMaxLength(max_length)
        textbox.setAlignment(Qt.AlignRight)
        textbox.setFont(QFont("Arial", 20))
        return textbox
    
    # create a general textbox
    def create_textbox(self):
        textbox = QLineEdit()
        textbox.setAlignment(Qt.AlignRight)
        textbox.setFont(QFont("Arial", 20))
        return textbox

    # create a general button
    def create_button(self, widget, text, function):
        button = QPushButton(widget)
        button.setText(text)
        button.clicked.connect(function)
        return button

    # create a general combo box
    def create_combo_box(self, items):
        comboBox = QComboBox()
        for item in items:
            comboBox.addItem(item)
        return comboBox

    # create a form layout
    def create_form_layout(self, inputs):
        layout = QFormLayout()
        for input_ in inputs:
            if type(input_) is tuple:
                layout.addRow(input_[0], input_[1])
            else:
                layout.addRow(input_)
        return layout

    def save_model_clicked(self):
        name_of_model = self.name_of_model_input.text()
        model = None
        if self.model_type_input.currentText() == "Binary":
            model = self.classifier.model
        else:
            model = self.classifier.model['model']
        self.classifier.save_model(name_of_model, model)
        labels = (self.labels_input.text()).split(" ")
        self.classifier.save_labels(labels, name_of_model)
        self.textEdit.append('Model Saved Successfully')

    def submit_clicked(self):
        if self.model_type_input.currentText() == 'Categorical':
            h_size = int(self.h_size_input.text())
            v_size = int(self.v_size_input.text())
            labels = (self.labels_input.text()).split(" ")
            epoch = int(self.epoch_input.text())
            batch_size = int(self.batch_size_input.text())
            self.classifier = img_classifier(True, labels, (h_size, v_size), epoch, batch_size)
            self.textEdit.append('Model Created. Press save to save the model to be used in the classifier application.')
        elif self.model_type_input.currentText() == "Binary":
            size = int(self.h_size_input.text())
            labels = (self.labels_input.text()).split(" ")
            epoch = int(self.epoch_input.text())
            self.classifier = bin_classifier(size, True, True, labels, epoch)
            self.textEdit.append('Model Created. Press save to save the model to be used in the classifier application.')
    
    def __init__(self, parent=None):
        super().__init__(parent)

        widget = QWidget()

        self.h_size_input = self.create_textbox_int(3)
        self.v_size_input = self.create_textbox_int(3)
        self.labels_input = self.create_textbox()
        self.epoch_input = self.create_textbox_int(1000)
        self.batch_size_input = self.create_textbox_int(1000)
        self.name_of_model_input = self.create_textbox()

        button = self.create_button(widget, "Save Model", self.save_model_clicked)
        
        self.button1 = self.create_button(widget, "Create Model", self.submit_clicked)

        self.model_type_input = self.create_combo_box(items=['Binary', 'Categorical'])

        self.textEdit = QTextEdit()

        flo = self.create_form_layout(inputs=[(self.model_type_input), ("Horizontal Image Size", self.h_size_input),
                                                ("Vertical Image Size", self.v_size_input), ("Enter All The Labels (w/ spaces in between)", self.labels_input),
                                                ("Enter the epoch (default 50)", self.epoch_input), ("Enter the batch size (default 10)", self.batch_size_input),
                                                (button, self.button1), (self.textEdit)])

        self.setLayout(flo)
        self.setWindowTitle("Image Classifier Model Creator")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Model_Creator()
    win.show()
    sys.exit(app.exec_())
