import matplotlib
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
import cv2

#from numba import jit, cuda 
#import pycuda.autoinit
#import pycuda.driver as cuda
#from pycuda.compiler import SourceModule

#Inspired by code that can be found at : https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/ 


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", default=None,
	help="path to output serialized model")
ap.add_argument("-e", "--epochs", type=int, default=30,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = ap.parse_args()

#from DataLoader import DataLoader
#Binary classification of colonoscopy videos 
# label 0 = no resection 
# label 1 = resection required


# function optimized to run on gpu  
#@jit(target ="cuda")
def main():
    

    train_data, train_labels = DataLoader("train", datapath=args.dataset, step=10) #dataset is path to data 
    val_data, val_labels = DataLoader("val", datapath=args.dataset, step=10)

    #train_data, train_labels = load_data("train", data_path=args.dataset)
    #val_data, val_labels = load_data("val", data_path=args.dataset)

    # load the ResNet-50 network, ensuring the head FC layer sets are left   off
    baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(576, 768, 3)))

    # construct the head of the model that will be placed on top of the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)    

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the training process
    for layer in baseModel.layers:
	    layer.trainable = False 

    print("[INFO] compiling model...")
    opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args.epochs)
    #binary crossentropy loss since we're doing a binary classification 
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])   

    val_freq = 2
    H = model.fit(x=train_data, y=train_labels, batch_size=60, epochs=args.epochs,
        validation_data=(val_data,val_labels), validation_freq=val_freq)

    print("finished fitting")
    # plot the training loss and accuracy
    
    N = args.epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N,val_freq), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N, val_freq), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    print("trying to plot")
    plt.show()

    '''
    with open('/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(H.history, file_pi)
    '''

    model.save("fulldata_10steps", save_format='h5')


def DataLoader(what, datapath, step):

    STEP = step
    frames = []
    labels = []
    f_counter = 0

    if what == "train":
        print("loading training data......")
        directory = datapath + 'train/'

    if what == "val":
        print("loading validation data.....")
        directory = datapath + 'val/'

    for filename in os.listdir(directory):

        video = cv2.VideoCapture(directory + filename)
        name = filename.split("_")[0] #get the class of the video
        if name=="adenoma" or name=="serrated":
            label = [0,1]
        elif name=='hyperplasic':
            label = [1,0] #one hot encoded labels 
        else :
            print("issue with filename " + filename)

        while True:
            has_frame, frame = video.read()

            if not has_frame:
                #print('Reached the end of the video')
                #print('selected frames:', f_counter // STEP)
                break

            if f_counter % STEP == 0:
                # cv2.imshow('frame', frame)
                # key = cv2.waitKey(50)
                #resized_frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                labels.append(label)

            f_counter += 1
    
    frames = np.asarray(frames)
    print(frames.shape)
    labels = np.asarray(labels)
    print(labels.shape)
    return frames, labels

if __name__ == "__main__":
    main()

