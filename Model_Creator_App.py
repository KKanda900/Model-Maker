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
        textbox = QLineEdit() # create the textbox widget
        textbox.setValidator(QIntValidator()) # make the textbox validate only ints
        textbox.setMaxLength(max_length) # set the max length depending on the argument
        textbox.setAlignment(Qt.AlignRight) # set the alignment of the textbox all the way to the right
        textbox.setFont(QFont("Arial", 20)) # set the font default to arial and 20
        return textbox # at the end return the textbox widget for use
    
    # create a general textbox
    def create_textbox(self):
        textbox = QLineEdit() # create a string textbox
        textbox.setAlignment(Qt.AlignRight) # align the textbox to the right
        textbox.setFont(QFont("Arial", 20)) # set the font to arial and 20 by default
        return textbox # return the textbox 

    # create a general button
    def create_button(self, widget, text, function):
        button = QPushButton(widget) # create the push button
        button.setText(text) # set the text of the button according to the text argument
        button.clicked.connect(function) # connect a function to the button depending on the function argument
        return button # return the button

    # create a general combo box
    def create_combo_box(self, items):
        comboBox = QComboBox() # create the combobox
        # iterate through all the items you want to add to the combobox
        for item in items:
            comboBox.addItem(item) # add the item to the combobox
        return comboBox # return the combobox

    # create a form layout (main layout of the application)
    def create_form_layout(self, inputs):
        layout = QFormLayout() # create the blank layout
        for input_ in inputs: # get all the inputs for the application
            # if the input is a tuple that means there should be a label then textbox/button 
            if type(input_) is tuple:
                layout.addRow(input_[0], input_[1]) # add those pieces
            else:
                # else just add the simple input
                layout.addRow(input_)
        return layout # return the layout

    # save the created model
    def save_model_clicked(self):
        name_of_model = self.name_of_model_input.text() # get the name of the model from the combobox
        model = None # set the model according to if its binary/categorical
        # check if the input is binary
        if self.model_type_input.currentText() == "Binary":
            model = self.classifier.model # make the model binary library
        else:
            model = self.classifier.model['model'] # else make it the categorical type library
        self.classifier.save_model(name_of_model, model) # save the model 
        labels = (self.labels_input.text()).split(" ") # get the labels from the label input
        self.classifier.save_labels(labels, name_of_model) # save the labels for later use in an application/another program
        self.textEdit.append('Model Saved Successfully') # write when the process of saving is done

    # create the model based on the specifications listed
    def submit_clicked(self):
        # check if the model type is categorical
        if self.model_type_input.currentText() == 'Categorical':
            h_size = int(self.h_size_input.text()) # create the horizontal size variable
            v_size = int(self.v_size_input.text()) # create the vertical size variable
            labels = (self.labels_input.text()).split(" ") # create the labels variable
            epoch = int(self.epoch_input.text()) # create the epoch variable
            batch_size = int(self.batch_size_input.text()) # create the batch size variable
            self.classifier = img_classifier(True, labels, (h_size, v_size), epoch, batch_size) # create the img classifier object (categorical)
            self.textEdit.append('Model Created. Press save to save the model to be used in the classifier application.') # write to the application when the model is done being created
        # check if the model type is binary
        elif self.model_type_input.currentText() == "Binary":
            size = int(self.h_size_input.text()) # size is the tuple of h_size and v_size
            labels = (self.labels_input.text()).split(" ") # create the labels variable
            epoch = int(self.epoch_input.text()) # create the epoch variabel
            self.classifier = bin_classifier(size, True, True, labels, epoch) # create the binary model maker object
            self.textEdit.append('Model Created. Press save to save the model to be used in the classifier application.') # write to the application that the model is being done created
    
    # when you initialize the object window, create these widgets
    def __init__(self, parent=None):
        super().__init__(parent)

        widget = QWidget() # create the widget instance

        self.h_size_input = self.create_textbox_int(3) # create the horizontal size input
        self.v_size_input = self.create_textbox_int(3) # create the vertical size input
        self.labels_input = self.create_textbox() # create the labels input
        self.epoch_input = self.create_textbox_int(1000) # create the epoch input
        self.batch_size_input = self.create_textbox_int(1000) # create the batch size input
        self.name_of_model_input = self.create_textbox() # create the name of model input 

        button = self.create_button(widget, "Save Model", self.save_model_clicked) # create the save model button
        
        self.button1 = self.create_button(widget, "Create Model", self.submit_clicked) # create the create model button

        self.model_type_input = self.create_combo_box(items=['Binary', 'Categorical']) # create the combobox binary/categorical

        self.textEdit = QTextEdit() # create the textEdit to write when model is saved/created

        # create the form layout with the valid arguments
        flo = self.create_form_layout(inputs=[(self.model_type_input), ("Horizontal Image Size", self.h_size_input),
                                                ("Vertical Image Size", self.v_size_input), ("Enter All The Labels (w/ spaces in between)", self.labels_input),
                                                ("Enter the epoch (default 50)", self.epoch_input), ("Enter the batch size (default 10)", self.batch_size_input),
                                                (button, self.button1), (self.textEdit)])

        self.setLayout(flo) # create the layout
        self.setWindowTitle("Image Classifier Model Creator") # after everything was created, write the name of the application

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Model_Creator()
    win.show()
    sys.exit(app.exec_())
