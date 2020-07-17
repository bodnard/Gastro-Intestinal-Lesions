# returns: [ (numpy.ndarray, str), ...]
#          [ (frame of a vid, label), ...]
# image shape: 576, 768, 3

import cv2
import time
import numpy as np
import os

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
        name = filename.split("_")[0]  # get the class of the video
        if name == "adenoma" or name == "serrated":
            label = [0, 1]
        elif name == 'hyperplasic':
            label = [1, 0]  # one hot encoded labels
        else:
            print("issue with filename " + filename)

        while True:
            has_frame, frame = video.read()

            if not has_frame:
                # print('Reached the end of the video')
                # print('selected frames:', f_counter // STEP)
                break

            if f_counter % STEP == 0:
                # cv2.imshow('frame', frame)
                # key = cv2.waitKey(50)
                # resized_frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                labels.append(label)

            f_counter += 1
    frames = np.asarray(frames)
    print(frames.shape)
    labels = np.asarray(labels)
    print(labels.shape)
    print("total frames:", f_counter)
    return frames, labels

startTime = time.time()

DataLoader("train", "data/", 10)
DataLoader("val", "data/", 10)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
