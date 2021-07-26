import os, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # dont show any tensorflow warning messages
import tensorflow as tf
from PIL import Image
import numpy as np

'''
Image_Classification Class

Description: 
1.  Used to process and train data to utilize to classify 
    different types of debris in any type of body of water. Using pre-defined libraries for 
    assistance.
2.  Uses binary classification as the model to predict.
3.  Uses a Convolutional Neural Network.
'''
class Image_Classification:

    # ---------------------  Tools used to train a model for classification -------------------------#
    
    train_data_gen = None # contains all the arrangements needed for the training data
    test_data_gen = None # contains all the arrangements needed for the testing data
    gen_training = None # fits the training data with the requirements 
    gen_testing = None # first the testing data with the requirements 

    labels = [] # contains the labels that the user wants to train based on
    img_size = 0 # holds the size of the image that the user wants to process on

    # model used for prediction thats created using the training data and testing data
    model = None

    training_samples = 50 # predefined training samples with how many images are in the training
    validation_samples = 30 # predefined testing samples with how many images are in testing
    batch_size = 1 # predefined batch size of how many images to take at one time

    # --------------------- Functions used for training the model ---------------------------------- #

    # resize the images in the testing/training directories so they can be utilized accurately
    def resize_images(self, directory_arr):
        # iterate through the testing and training directory
        while len(directory_arr) != 0:
            f = directory_arr.pop(0) # pop the first directory in the list to go through
            for file in os.listdir(f):
                f_img = f+"/"+file # for opening purposes
                # open the image then resize it
                img = Image.open(f_img)
                img = img.resize((self.img_size, self.img_size))
                # after resizing save the image
                img.save(f_img)

    # define the labels and images depending on the directory path
    def set_data(self, directory_path):
        data_labels = [] # define the set of labels according to the name of the file
        data_images = [] # define the images
        # iterate through all the images in the directory
        for filename in os.listdir(directory_path): 
            if filename.split('.')[1] == 'jpg': # all the images should be defined as a jpg
                # Get the values of the images at the directory path
                img = cv2.imread(os.path.join(directory_path,filename))
                
                # Spliting file names and storing the labels for image in list
                data_labels.append(filename.split('_')[0])
                
                # Resize all images to a specific shape
                img = cv2.resize(img,self.shape)
                
                data_images.append(img) # append the image
        data_labels = pd.get_dummies(data_labels).values # Get the categorical data
        data_images = np.array(data_images) # Define the image array as a np array for fitting

        return data_labels, data_images # return the labels, images for the specific directory
        
    # initialize the tools for training the model and used to create the model
    def __init__(self, img_size, resize, create_model, labels, epoch):
        self.img_size = img_size  # set the image size to the decided size for training
        
        # resize the images if they weren't already otherwise skip the conditional
        if resize == True:
            self.resize_images(directory_arr=['./Binary_Classification/Input/Train/Plastic', './Binary_Classification/Input/Test/Plastic', './Binary_Classification/Input/Test/Not_Plastic', './Binary_Classification/Input/Train/Not_Plastic'])

        # define the arguments for preprocessing the training dataset
        self.train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

        # define the arguments for preprocessing the testing dataset
        self.test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        # define the training dataset with the arguments given by train_data_gen
        self.gen_training = self.train_data_gen.flow_from_directory('./Binary_Classification/Input/Train', target_size=(self.img_size, self.img_size), batch_size=self.batch_size, class_mode='binary')
        
        # define the testing dataset with the arguments given by the test_data_gen
        self.gen_testing = self.test_data_gen.flow_from_directory('./Binary_Classification/Input/Test', target_size=(self.img_size, self.img_size), batch_size=self.batch_size, class_mode='binary')

        # create a model if it wasn't already used, if create_model == False then the Object is used not for the model
        if create_model == True:
            self.labels = labels  # obtain the user defined labels
            self.model = self.create_model(epoch) # create the model

    # create the model used for predicition using a Convolutional Neural Network Model (9 layer NN) -> in order to get an accurate model for testing
    def create_model(self, epoch):
        model = tf.keras.models.Sequential() # define the model to be sequential which is best for RGB image data

        # define the first set of layers: 1 Conv2D, 1 Activation, 1 MaxPooling
        model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(self.img_size, self.img_size, 3))) # define the first layer
        model.add(tf.keras.layers.Activation('relu')) # define the second layer
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) # define the third layer

        # define the second set of layers: 1 Conv2D, 1 Activation, 1 MaxPooling
        model.add(tf.keras.layers.Conv2D(32, (3, 3))) # define the fourth layer
        model.add(tf.keras.layers.Activation('relu')) # define the fifth layer
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) # define the sixth layer

        # define the third set of layers: 1 Conv2D, 1 Activation, 1 MaxPooling
        model.add(tf.keras.layers.Conv2D(64, (3, 3))) # define the seventh layer
        model.add(tf.keras.layers.Activation('relu')) # define the eight layer
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) # define the ninth layer

        # define the fourth set of layers: flatten the data, add a dense layer, activation, add dropout, add another dense layer and top it off with the activation
        model.add(tf.keras.layers.Flatten()) # define the tenth layer
        model.add(tf.keras.layers.Dense(64)) # define the eleventh layer
        model.add(tf.keras.layers.Activation('relu')) # define the twelfth layer
        model.add(tf.keras.layers.Dropout(0.5))  # define the thirteenth layer
        model.add(tf.keras.layers.Dense(1)) # define the fourteenth layer 
        model.add(tf.keras.layers.Activation('sigmoid')) # define the fiftheenth layer

        # compile the model with certain attributes, i.e: loss function, optimizer, metrics, ....
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        
        # train the model utilizing the training, defined epochs, validation data, ....
        model.fit(self.gen_training, steps_per_epoch=self.training_samples//self.batch_size, epochs=epoch, validation_data=self.gen_testing, validation_steps=self.validation_samples//self.batch_size)
        
        return model # return the model at the end
    
    # --------------------- Functions used for predicting based on the preprocessed directory ---------------------------------- #

    # utilizing the created model from the function create_model() predict if a patch of images are affiliated with Plastic
    def predict(self, model, preprocessed_directory):
        predictions = model.predict(preprocessed_directory) # make sure you are using a preprocessed directory that contains pixel values
        pred_classes = predictions.argmax(axis=-1) # find the class each belongs to using the argmax function
        return predictions, pred_classes  # return the predictions

    # using the predicition data from the predict function, classify the model using the defined labels above
    def classify(self, predicitions):
        class_arr = [] # contains the dictionary that corresponds to what each image in a directory corresponds too

        # make the association based on the percentage from predicition to the classes
        for i in range(len(predicitions)):
            percentage = float(predicitions[i]) # conver the predicitions to float values
            if percentage >= .50: # well established treshold to affiliate the image with the 0 class label
                classification_dictionary = {} # define a dictionary to hold the attributes of the image
                classification_dictionary['Image #'] = i # i represents the image in the directory
                classification_dictionary['Label'] = self.labels[0] # define the label
                class_arr.append(classification_dictionary) # append it to the class array 
            else:
                classification_dictionary = {} # define a dictionary to hold the attributes of the image
                classification_dictionary['Image #'] = i # i represents the image in the dictionary
                classification_dictionary['Label'] = self.labels[1] # define the label
                class_arr.append(classification_dictionary) # append it to the class array

        return class_arr # return the class array used for classification
    
    # ------------------------------------------------- Extra utility for preprocessing and saving the created model ---------------------------------- #

    # preprocess any given directory that has the subdirectories affiliated with the labels
    def preprocess_directory(self, directory_path):
        app_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # define the attributes of defining the directory
        preprocessed_directory = app_data_gen.flow_from_directory(directory_path, target_size=(self.img_size, self.img_size), batch_size=self.batch_size, class_mode='binary') # preprocess the directory
        return preprocessed_directory # return the preprocessed directory predicitions

    # save any model used to be utilized for later
    def save_model(self, name_of_model, model):
        model.save('./Models/'+name_of_model+'.h5') # save the model in the models directory

