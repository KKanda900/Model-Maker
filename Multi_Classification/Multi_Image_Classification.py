# Primary Python Files for Image Classification
import numpy as np 
import pandas as pd 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # dont show any tensorflow warning messages
import matplotlib.pyplot as plt
import cv2, dill

# Keras libraries used for making the model
import keras
from keras.utils import to_categorical
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
from keras.models import Sequential

# Sklearn library for splitting the data precisely
from sklearn.model_selection import train_test_split

'''
Multi_Image_Classification Class

Description: 
1.  Identify different sets of images based on the labels you provide.
2.  Works based off a sequential model.
3.  Uses a Convolutional Neural Network.
'''
class Multi_Image_Classification:

    # ------------------------------ Generic Fields Needed for Training ---------------------------------- #
    shape = (200,200) # predefine a established shape for training and resizing the images (default)
    labels = [] # define the labels to train on
    
    # --------------------------- Training Tools ---------------------------------- #
    train_path = './Multi_Classification/train' # define the path where the training images are located
    train_labels = None # define the labels (same as testing)
    train_images = None # define the images with the training 
    x_train = None # split the training images for training
    y_train = None # split the training labels for training
    
    # ------------------------- Testing Tools -------------------------------------- #
    test_path = './Multi_Classification/test' # define the path where the testing images are located
    x_val = None # split the training images for testing
    y_val = None # split the training labels for testing
    test_labels = None # define the testing labels (same as training)
    test_images = None # define the testing images 
    
    # ----------------------------------- Main Model Tools ------------------------------- #
    epoch = 50 # default epoch 
    batch_size = 10 # default batch size
    model = None # define the model (Sequential for Image Classification)

    # ------------------------- Define the Functions for Making the model ---------------------- #

    # define the labels and images depending on the directory path
    def set_data(self, directory_path):
        data_labels = [] # define the set of labels according to the name of the file
        data_images = [] # define the images
        
        # iterate through all the images in the directory
        for filename in os.listdir(directory_path): 
            # Get the values of the images at the directory path
            img = cv2.imread(os.path.join(directory_path, filename))
            # Spliting file names and storing the labels for image in list
            data_labels.append(filename.split('_')[0])
            # Resize all images to a specific shape
            img = cv2.resize(img, self.shape)
            data_images.append(img)  # append the image
        
        data_labels = pd.get_dummies(data_labels).values # Get the categorical data
        data_images = np.array(data_images) # Define the image array as a np array for fitting

        return data_labels, data_images # return the labels, images for the specific directory

    # define the tools for utilzing on creation of the object
    def __init__(self, create_model, labels, shape, epoch, batch_size):
        np.random.seed(1) # sets the random seed of the NumPy pseudo-random number generator

        self.shape = shape # let the user enter the shape of the images to be formed (default 200x200)

        # let the user define the labels for their model they want to create
        self.labels = labels # default values

        # define the training images and labels
        self.train_labels, self.train_images = self.set_data(self.train_path) 

        # Splitting Training data into train and validation dataset
        self.x_train,self.x_val,self.y_train,self.y_val = train_test_split(self.train_images,self.train_labels,random_state=1)
        
        # define the test labels and images
        self.test_labels, self.test_images = self.set_data(self.test_path)
        
        # define the model for predicition 
        if create_model == True:
            self.model = self.create_model(epoch, batch_size, self.x_train, self.y_train, self.x_val, self.y_val)

    # create the model to be used for predicition
    def create_model(self, epoch, batch_size, x_train, y_train, x_val, y_val):
        model = Sequential() # define the model as sequential
        
        model.add(Conv2D(kernel_size=(3,3), filters=32, activation='tanh', input_shape=(200,200,3,))) # define the first layer
        model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh')) # define the second layer
        model.add(MaxPool2D(2,2)) # define the third layer
        model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh')) # define the fourth layer
        model.add(MaxPool2D(2,2)) # define the fifth layer
        model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh')) # define the sixth layer
        model.add(Flatten()) # define the seventh layer
        model.add(Dense(20,activation='relu')) # define the eigth layer
        model.add(Dense(15,activation='relu')) # define the ninth layer
        model.add(Dense(len(self.labels),activation = 'softmax')) # define the tenth layer (according to the number of labels for the model)
            
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam') # compile the models with categorical because we are working with multiple labels
        history = model.fit(x_train,y_train,epochs=epoch,batch_size=batch_size,validation_data=(x_val,y_val)) # train the model
        
        # after the training is done, define a dictionary that holds the model and history from the training
        complete_model = {} # define the dictionary
        complete_model['model'] = model # define the model with its key
        complete_model['history'] = history # define the history with its key
        complete_model['labels'] = self.labels # save the labels into the dictionary
        
        return complete_model # return the model at the end

    # function to save the model that was created in the create_model function
    def save_model(self, model_name, model):
        model.save('./Models/{}.h5'.format(model_name)) # save the model in the models directory

    # function to save the model's labels to be used later
    def save_labels(self, labels, model_name):
        f = open('./Models/{}_Labels.txt'.format(model_name), 'a') # create the .txt file that will contain the labels of the model
        # iterate through the labels when the model was first created
        for i in range(len(labels)):
            f.write("{}\n".format(labels[i])) # write the labels to the file
        f.close() # after iterating through all the labels, close the file so the space can be free

    # ------------------------------------------------------ Define the functions used for classifiying --------------------------------------------- #
    
    # classifies images based on the model and the selected image
    def classify_image(self, image, model):
        
        checkImage = image[0]
        checklabel = image[0]

        predict = model.predict(np.array(checkImage))
        predicted_label = self.labels[np.argmax(predict)]
                
        return predicted_label # return the predicted label from the labels provided by the user


