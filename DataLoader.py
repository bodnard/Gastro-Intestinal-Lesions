
import cv2 as cv

STEP = 5

def DataLoader(what, datapath):

    frames = []
    f_counter = 0

    if what == "train":
        video = cv.VideoCapture('data/train/adenoma_01_WL.mp4')

    if what == "val":
        video = cv.VideoCapture('data/val/adenoma_10_WL.mp4')

    while True:
        has_frame, frame = video.read()

        if not has_frame:
            print('Reached the end of the video')
            print('selected frames:', f_counter // STEP)
            return frames

        if f_counter % STEP == 0:
            # cv2.imshow('frame', frame)
            # key = cv2.waitKey(50)
            frames.append(frame)

        f_counter += 1
