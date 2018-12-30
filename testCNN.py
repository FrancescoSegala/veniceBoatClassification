import tensorflow as tf
import keras 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import classification_report


IMG_HEIGHT, IMG_WIDTH = 180,200
CHANNELS = 3
NUM_CLASSES = 24 
N_TRAIN = 4774
N_TEST = 1672
epochs = 50
batch_size = 32
train_dir = "../data/sc5"
aux_test = "../data/sc5-test"
test_dir = "../data/sc5-2013-Mar-Apr-Test-20130412"


#################################### DATA PREPARATION #####################################
#from 
#https://keras.io/preprocessing/image/
#and 
#https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d 


#this function {prepare_test_set} is a bootstrap method to manimpulate the data we have from the test set in order to 
#have a format that is the same as what we have for the training 
#so we will copy all the images in the test set in a directory called {aux_test} where each 
#subdirectory contains the test images for the specific class. It not take any parameter because is experiment specific.
def prepare_test_set():
        d = {}
        #in train_list there is the list of all classes belonging to the experiment 
        train_list = [x[0].split("/")[-1] for x in os.walk(train_dir)]
        train_list.remove(train_dir.split("/")[-1])
        #create the auxiliary test folder 
        if not os.path.exists(aux_test):
                os.makedirs(aux_test)
        with open(test_dir+"/ground_truth.txt", "r") as file:
                lines = file.readlines()
                for line in lines: 
                        #this because the ground_truth.txt file has different label wrt the class name.
                        line = line.split(";")
                        line[1] = line[1].strip()
                        line[1] = line[1].replace("Snapshot Acqua", "Water")
                        line[1] = line[1].replace(" ", "")
                        line[1] = line[1].replace(":", "")
                        #skip the partial view of a boat and also the aggregation of some of them 
                        if line[1].__contains__("Snapshot Barca"):
                                continue        
                        #this try-catch is for keyError when there is no key in the dict 
                        try:
                                d[line[1]] += [line[0]]
                        except KeyError as error:
                                d[line[1]] = [line[0]]
                #creation of the aux directory tree where at each class correspond a directory conaining the inages                 
                for folder in train_list:
                        if not os.path.exists(aux_test+"/"+folder):
                                os.makedirs(aux_test+"/"+folder) 
                                try:
                                        for filename in d[folder]:
                                                copyfile(test_dir+"/"+filename, aux_test+"/"+folder+"/"+filename)
                                except KeyError as error:
                                        #we leave an empty directory
                                        pass


def get_generators():
        #simple wrapper, avoids floating code
        train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=batch_size,
                class_mode='categorical')

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        validation_generator = test_datagen.flow_from_directory(
                directory=aux_test,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=batch_size)
        return train_generator, validation_generator
        

##############################################################################################################        

def leNet(optimizer="adam"):
    
    # A Keras tensor is a tensor object from the underlying backend (Theano or TensorFlow),
    # which we augment with certain attributes that allow us to build a Keras model just by 
    # knowing the inputs and outputs of the model.
    
    #input layer    
    #tf.keras.Input() is used to instantiate a Keras tensor 
    image_input = tf.keras.Input( shape=(IMG_WIDTH,IMG_HEIGHT,CHANNELS), name="input_layer" )
     
    
    #two convolutional layers
    #here filters should represent the dimensionality of the output space
    conv_layer_1 = tf.keras.layers.Conv2D( filters = 32 , kernel_size = (6 , 6), padding="same", activation="relu" )(image_input)
    conv_layer_1 = tf.keras.layers.MaxPooling2D(padding='same')(conv_layer_1)

    conv_layer_2 = tf.keras.layers.Conv2D( filters = 32 , kernel_size = (6 , 6), padding="same", activation="relu" )(conv_layer_1)
    conv_layer_2 = tf.keras.layers.MaxPooling2D(padding="same")(conv_layer_2) 
    
    conv_layer_3 = tf.keras.layers.Conv2D( filters = 32 , kernel_size = (6 , 6), padding="same", activation="relu" )(conv_layer_2)
    conv_layer_3 = tf.keras.layers.MaxPooling2D(padding="same")(conv_layer_3)     


    #Flatten the output of the convolutional layers
    conv_flat = tf.keras.layers.Flatten()(conv_layer_2)
    
    #two dense layers 
    #units param model the dimensionality of the output space 
    #Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time
    # which helps prevent overfitting
    dense_layer_1 = tf.keras.layers.Dense(units=64, activation='relu')(conv_flat)
    dense_layer_1 = tf.keras.layers.Dropout(0.5)(dense_layer_1)
  
    
    #one output layer
    output_layer = tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax", name="boat_type")(dense_layer_1)
    if optimizer=="gradient":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-3 )

    model = tf.keras.Model( inputs=image_input, outputs=[output_layer] )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model 


#The structure of this network is the same as the one we saw in the excerxise during the lectures.
def alexNet(optimizer="adam", loss="categorical_crossentropy" ):
    
    image_input = tf.keras.Input( shape=(IMG_WIDTH,IMG_HEIGHT,CHANNELS), name="Boat_input"  )
    
    #convolutional layer 1 
    conv_layer_1 = tf.keras.layers.Conv2D( filters = 96 , kernel_size = (11 , 11), 
            strides=(4,4), padding="valid", activation="relu" )(image_input)

    conv_layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv_layer_1)

    conv_layer_1 = tf.keras.layers.BatchNormalization()(conv_layer_1)

    #convolutional layer 2 
    conv_layer_2 = tf.keras.layers.Conv2D( filters = 256 , kernel_size = (6, 6), 
            strides=(1,1), padding="valid", activation="relu" )(conv_layer_1)

    conv_layer_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv_layer_2)

    conv_layer_2 = tf.keras.layers.BatchNormalization()(conv_layer_2)

    #convolutional layer 3
    conv_layer_3 = tf.keras.layers.Conv2D( filters = 384 , kernel_size = (3 , 3), 
            strides=(1,1), padding="valid", activation="relu" )(conv_layer_2)

    conv_layer_3 = tf.keras.layers.BatchNormalization()(conv_layer_3)

    #convolutional layer 4
    conv_layer_4 = tf.keras.layers.Conv2D( filters = 384 , kernel_size = (3 , 3), 
            strides=(1,1), padding="valid", activation="relu" )(conv_layer_3)
            
    conv_layer_4 = tf.keras.layers.BatchNormalization()(conv_layer_4)

    #convolutional layer 5 
    conv_layer_5 = tf.keras.layers.Conv2D( filters = 256 , kernel_size = (3 , 3), 
            strides=(1,1), padding="valid", activation="relu" )(conv_layer_4)

    conv_layer_5 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv_layer_5)

    conv_layer_5 = tf.keras.layers.BatchNormalization()(conv_layer_5)

    #Flatten the output of the convolutional layers
    conv_flat = tf.keras.layers.Flatten()(conv_layer_5)
    
    #dense layers 
    dense_layer_1 = tf.keras.layers.Dense(units=4096, activation='relu')(conv_flat)
    dense_layer_1 = tf.keras.layers.Dropout(0.4)(dense_layer_1)
    dense_layer_1 = tf.keras.layers.BatchNormalization()(dense_layer_1)

    dense_layer_2 = tf.keras.layers.Dense(units=4096, activation='relu')(dense_layer_1)
    dense_layer_2 = tf.keras.layers.Dropout(0.4)(dense_layer_2)
    dense_layer_2 = tf.keras.layers.BatchNormalization()(dense_layer_2)

    dense_layer_3 = tf.keras.layers.Dense(units=1000, activation='relu')(dense_layer_2)
    dense_layer_3 = tf.keras.layers.Dropout(0.4)(dense_layer_3)
    dense_layer_3 = tf.keras.layers.BatchNormalization()(dense_layer_3)

    #output layer 
    output_layer = tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax", name="Boat_type")(dense_layer_3)

    model = tf.keras.Model( inputs=image_input, outputs=[output_layer] )
    #default adam optimizer has learning rate of 1e-3, in addition you can create another learning rate optimizer via Adam(lr=1e-3)
    if optimizer=="gradient":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-3 )
    if loss=="mse":
        loss="mse"
    model.compile(loss=loss, optimizer=optimizer, metrics=["mse", 'accuracy'])

    return model 

#The structure of this network is the same as the one we saw in the excerxise during the lectures.
def alexNet(optimizer="adam", loss="categorical_crossentropy" ):
    
    image_input = tf.keras.Input( shape=(IMG_WIDTH,IMG_HEIGHT,CHANNELS), name="Boat_input"  )
    
    #convolutional layer 1 
    conv_layer_1 = tf.keras.layers.Conv2D( filters = 96 , kernel_size = (11 , 11), 
            strides=(4,4), padding="valid", activation="relu" )(image_input)

    conv_layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv_layer_1)

    conv_layer_1 = tf.keras.layers.BatchNormalization()(conv_layer_1)

    #convolutional layer 2 
    conv_layer_2 = tf.keras.layers.Conv2D( filters = 256 , kernel_size = (6, 6), 
            strides=(1,1), padding="valid", activation="relu" )(conv_layer_1)

    conv_layer_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv_layer_2)

    conv_layer_2 = tf.keras.layers.BatchNormalization()(conv_layer_2)

    #convolutional layer 3
    conv_layer_3 = tf.keras.layers.Conv2D( filters = 384 , kernel_size = (3 , 3), 
            strides=(1,1), padding="valid", activation="relu" )(conv_layer_2)

    conv_layer_3 = tf.keras.layers.BatchNormalization()(conv_layer_3)

    #convolutional layer 4
    conv_layer_4 = tf.keras.layers.Conv2D( filters = 384 , kernel_size = (3 , 3), 
            strides=(1,1), padding="valid", activation="relu" )(conv_layer_3)
            
    conv_layer_4 = tf.keras.layers.BatchNormalization()(conv_layer_4)

    #convolutional layer 5 
    conv_layer_5 = tf.keras.layers.Conv2D( filters = 256 , kernel_size = (3 , 3), 
            strides=(1,1), padding="valid", activation="relu" )(conv_layer_4)

    conv_layer_5 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv_layer_5)

    conv_layer_5 = tf.keras.layers.BatchNormalization()(conv_layer_5)

    #Flatten the output of the convolutional layers
    conv_flat = tf.keras.layers.Flatten()(conv_layer_5)
    
    #dense layers 
    dense_layer_1 = tf.keras.layers.Dense(units=4096, activation='relu')(conv_flat)
    dense_layer_1 = tf.keras.layers.Dropout(0.4)(dense_layer_1)
    dense_layer_1 = tf.keras.layers.BatchNormalization()(dense_layer_1)

    dense_layer_2 = tf.keras.layers.Dense(units=4096, activation='relu')(dense_layer_1)
    dense_layer_2 = tf.keras.layers.Dropout(0.4)(dense_layer_2)
    dense_layer_2 = tf.keras.layers.BatchNormalization()(dense_layer_2)

    dense_layer_3 = tf.keras.layers.Dense(units=1000, activation='relu')(dense_layer_2)
    dense_layer_3 = tf.keras.layers.Dropout(0.4)(dense_layer_3)
    dense_layer_3 = tf.keras.layers.BatchNormalization()(dense_layer_3)

    #output layer 
    output_layer = tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax", name="Boat_type")(dense_layer_3)

    model = tf.keras.Model( inputs=image_input, outputs=[output_layer] )
    #default adam optimizer has learning rate of 1e-3, in addition you can create another learning rate optimizer via Adam(lr=1e-3)
    if optimizer=="gradient":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-3 )
    if loss=="mse":
        loss="mse"
    model.compile(loss=loss, optimizer=optimizer, metrics=["mse", 'accuracy'])

    return model 
######################################### MAIN PROGRAM ####################################################
def main(Args):
        #data preparation
        prepare_test_set()
        train_generator , validation_generator = get_generators()
        
        #save the weights 
        model = "" 
        if Args.cnn == "leNet":
                print("model: leNet")
                model = leNet()
        else: 
                print("model: alexNet")
                model = alexNet()

        model.load_weights(Args.model)
        print("model loaded ")
        print(model.inputs)
        print("CNN..")
        print(model.outputs)
        
        #evaluate results 
        print("computing the score...")
        evaluate = model.evaluate(validation_generator)
        i=0
        for metric_name in model.metrics_names:
                print("Test ", metric_name, ": ", evaluate[i])        
                i+=1
        
        #precision recall f1 score 
        predict = model.predict_generator(validation_generator)
        y_pred = np.argmax(predict, axis=-1)
        y_true = validation_generator.classes
        print(classification_report(y_true, y_pred) )
        




###################################################### AUX FUNCTIONS #######################################################################


def ParseArgs():
    Args = argparse.ArgumentParser(description="testing module" )
    Args.add_argument("--model", required=True,help="the model.h5 complete path wrt the current position")
    Args.add_argument("--cnn", default= "alexNet",help="the model to use")
    return Args.parse_args()


#######################################################################################################################################

if __name__ == "__main__":
       main(ParseArgs())
