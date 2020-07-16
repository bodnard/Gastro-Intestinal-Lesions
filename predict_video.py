from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
#ap.add_argument("-i", "--input", required=True,
#	help="path to our input video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = ap.parse_args()


def main():
  
    model = load_model("D:/Gastro-Intestinal-Lesions/models/" + args.model)
    print("model loaded")

    #Q = deque(maxlen=args["size"])
    # initialize the vid  model = load_model(args.model)eo stream, pointer to output video file, and
    # frame dimensions
    directory = "data/test/"
    for filename in os.listdir(directory):

        print(filename)
        predictions = []

        name = filename.split("_")[0] #get the class of the video
        if name=="adenoma" or name=="serrated":
            label = "resection [1]"
        elif name=='hyperplasic':
            label = "no resection [0]" #one hot encoded labels 
        else :
            print("issue with filename " + filename)

        vs = cv2.VideoCapture(directory + filename)
        writer = None
        (W, H) = (None, None)
        # loop over frames from the video file stream
        while True:
            # read the next frame from the file
            (grabbed, frame) = vs.read()
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

        # make predictions on the frame and then update the predictions
	    # queue
            preds = model.predict(np.expand_dims(frame, axis=0))[0]
            predictions.append(preds)

    # perform prediction averaging over the current history of
	# previous predictions
        results = np.array(predictions).mean(axis=0)
        i = np.argmax(results)
        print("Real = " + label)
        print("Prediction = " + str(i))

if __name__ == "__main__":
    main()
