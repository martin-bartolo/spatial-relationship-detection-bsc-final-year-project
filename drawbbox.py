import cv2
import numpy as np
import tensorflow_yolov4_tflite.core.utils as utils
from absl import app, flags, logging

def main(_argv):
    pred_bbox = [[[[0.5532611608505249, 0.4249439537525177, 0.6599859595298767, 0.4597399830818176]]], [[0.555]], [[14.0]], [1]]
    #draw bounding boxes
    frame = cv2.imread("C:/Users/martin/Desktop/image_1648.jpg")
    result = utils.draw_bbox(frame, pred_bbox)

    #display frame with all detected objects
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result", result)
    cv2.waitKey(0)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass