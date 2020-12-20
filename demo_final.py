import time

import cv2

import simpleaudio as sa
import os.path

import numpy as np
# import the models for further classification experiments
from tensorflow.keras.applications import (
    vgg16
)

# init the models
vgg_model = vgg16.VGG16(weights='imagenet')


import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# # 1357
# import glob
# import cv2
# file_path = ''
# for i in range(1, 1358):
#     im = cv2.imread(file_path + '0'*(4-len(str(i))) + str(i) + '.jpg')
#     cv2.imshow('image', im)
#     cv2.waitKey(10)

# video_name = "demo_video"
image_path = "output/"
# cap = cv2.VideoCapture(video_name + '.mp4')


label_dict = {'bee_eater':'bird', 'goldfinch':'bird', 'jay':'bird', 'black_grouse':'bird', 'magpie':'bird', 'indigo_bunting':'bird', 'hummingbird':'bird', 'ox':'cow', 'sorrel':'cow', 'Siberian_husky':'dog', 'malamute':'dog', 'Norwegian_elkhound':'dog'}

# fps = cap.get(cv2.CAP_PROP_FPS)


start_time = time.time()

frame_number = 0
flag = True
for i in range(1, 1358):
    if os.path.isfile(image_path + '0' * (4 - len(str(i))) + str(i) + '.jpg'):
        im = cv2.imread(image_path + '0' * (4 - len(str(i))) + str(i) + '.jpg')
    else:
        continue
    # Capture frames in the video
    # ret, frame = cap.read()
    # frame_number = cap.get(1) - 1
    now = time.time()
    end_time = time.time()
    if flag == True or end_time - start_time > 1:

        start_time = end_time

        frame_number += 28

        timeDiff = time.time() - now
        # print(timeDiff)

    now = time.time()


    # height, width, layers = frame.shape
    # new_h = height / 2
    # new_w = width / 2
    # resize = cv2.resize(frame, (int(new_w), int(new_h)))

    # print((1.0 - timeDiff)/fps)
    # print(time.time() - now)
    time.sleep((1.0 - timeDiff)/30)
    # Display the resulting frame
    # cv2.imshow('video', resize)
    # print(cap.get(1)-1)
    # cv2.imwrite("output/%04d.jpg" % int(cap.get(1) - 1), resize)
    # timeDiff = time.time() - now
    # if (timeDiff < 1.0/(fps)):
    #     print(1.0/(fps) - timeDiff)
    #     # time.sleep(1.0/(fps) - timeDiff)

    cv2.imshow('image', im)
    # creating 'q' as the quit
    # button for the video
    cv2.waitKey(10)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    if flag == True:
        wave_obj = sa.WaveObject.from_wave_file("demo_audio.wav")
        wave_obj.play()
        start_time = time.time()
        flag = False

# release the cap object
# cap.release()
# close all windows
cv2.destroyAllWindows()



