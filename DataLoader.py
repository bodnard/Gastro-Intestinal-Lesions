# returns: [ (numpy.ndarray, str), ...]
#          [ (frame of a vid, label), ...]
# image shape: 576, 768, 3

import cv2
import time

def DataLoader(what, datapath, step):

    STEP = step
    directory = datapath
    
    frames = []
    f_labels = []  # the labels for each frame
    frame_label_list = []
    f_counter = 0

    if what == "train":
        directory += "train/"

    if what == "val":
        directory += "val/"
    
    labels = open(directory + "labels.txt")
    video_labels = labels.read().splitlines()
    
    names = open(directory + "names.txt")
    video_names = names.read().splitlines()
    # print(video_names)
    
        for index, video_name in enumerate(video_names):
        video = cv2.VideoCapture(directory + video_name)
        vid_set = len(video_names)
        while True:
            has_frame, frame = video.read()

            if not has_frame:
                # print('Reached the end of ', index + 1, '/', vid_set, ' video')
                print('fram shape:', frames[-1].shape)
                # print('selected frames:', f_counter // STEP)
                break

            if f_counter % STEP == 0:
                # cv2.imshow('frame', frame)
                # key = cv2.waitKey(50)
                frames.append(frame)
                f_labels.append(video_labels[index])
                frame_label_list.append((frames[-1], f_labels[-1]))

            f_counter += 1
    print("selected frames:", f_counter // STEP)
    # cv2.destroyAllWindows()
    return frame_label_list


startTime = time.time()

DataLoader("train", "data/", 10)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
