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

#Inspired by code that can be found at : https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/ 


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", default=None,
	help="path to output serialized model")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = ap.parse_args()

from DataLoader import DataLoader
#Binary classification of colonoscopy videos 
# label 0 = no resection 
# label 1 = resection required


def main():
    

    train_data, train_labels = DataLoader("train", datapath=args.dataset) #dataset is path to data 
    val_data, val_labels = DataLoader("val", datapath=args.dataset)

    # load the ResNet-50 network, ensuring the head FC layer sets are left   off
    baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

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

    H = model.fit(x=train_data, y=train_labels, batch_size=None, epochs=args.epochs,
        validation_data=(val_data,val_labels))

    # plot the training loss and accuracy
    N = args.epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")







def evaluate(model):
    print("Evaluating Network......")
    #evaluate one video at a time 
    #prediction done from all frames of that image
    accuracy = []
    model.eval()
    #for video in validation set

        #for frame in the video 
        #predict the frame, 
        #combine predictions 


    #check accuracy vs - best accuracy :
        #save model weights 

















if __name__ == "__main__":
    main()


