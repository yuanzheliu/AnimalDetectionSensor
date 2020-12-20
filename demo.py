import time

import cv2

import simpleaudio as sa

import numpy as np
# import the models for further classification experiments
from tensorflow.keras.applications import (
    vgg16
)

# init the models
vgg_model = vgg16.VGG16(weights='imagenet')

# inception_model = inception_v3.InceptionV3(weights='imagenet')
#
# resnet_model = resnet50.ResNet50(weights='imagenet')
#
# mobilenet_model = mobilenet.MobileNet(weights='imagenet')

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions

video_name = "demo_video"
cap = cv2.VideoCapture(video_name + '.mp4')


label_dict = {'bee_eater':'bird', 'goldfinch':'bird', 'jay':'bird', 'black_grouse':'bird', 'magpie':'bird', 'indigo_bunting':'bird', 'hummingbird':'bird', 'ox':'cow', 'sorrel':'cow', 'Siberian_husky':'dog', 'malamute':'dog', 'Norwegian_elkhound':'dog'}
#
# data = pd.read_csv('class_pred.txt', sep=",", header=None)
# data.columns = ["video_name", "frame_number", "class_label", "bbox_0", "bbox_1", "bbox_2", "bbox_3", "confidence_score"]
# data = data[data['video_name'] == video_name]

fps = cap.get(cv2.CAP_PROP_FPS)


start_time = time.time()

frame_number = 0
flag = True
while (True):
    # Capture frames in the video
    ret, frame = cap.read()
    # frame_number = cap.get(1) - 1
    now = time.time()
    end_time = time.time()
    if flag == True or end_time - start_time > 1:

        start_time = end_time

        # print(cap.get(1) - 1)
        cap.set(1, frame_number)
        ret, frame = cap.read()

        ### ImageNet Prediction. ###
        # load an image in PIL format
        original = cv2.resize(frame, (224, 224))
        # print('PIL image size', original.size)
        # plt.imshow(original)
        # plt.show()

        # convert the PIL image to a numpy array
        # IN PIL - image is in (width, height, channel)
        # In Numpy - image is in (height, width, channel)
        numpy_image = img_to_array(original)
        # plt.imshow(np.uint8(numpy_image))
        # plt.show()
        # print('numpy array size', numpy_image.shape)

        # Convert the image / images into batch format
        # expand_dims will add an extra dimension to the data at a particular axis
        # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
        # Thus we add the extra dimension to the axis 0.
        image_batch = np.expand_dims(numpy_image, axis=0)
        # print('image batch size', image_bau

        # prepare the image for the VGG model
        processed_image = vgg16.preprocess_input(image_batch.copy())

        # get the predicted probabilities for each class
        predictions = vgg_model.predict(processed_image)
        # print predictions
        # convert the probabilities to class labels
        # we will get top 5 predictions which is the default
        label_vgg = decode_predictions(predictions)
        # # print VGG16 predictions
        # for prediction_id in range(len(label_vgg[0])):
        #     print(label_vgg[0][0])

        # frame_data = data[data['frame_number'] == frame_number]
        # class_label = frame_data['class_label'].iloc[0]
        # confidence_score = frame_data['confidence_score'].iloc[0]
        class_label = label_vgg[0][0][1]
        confidence_score = label_vgg[0][0][2]

        if class_label in label_dict:
            class_label = label_dict[class_label]

        frame_number += 28

        timeDiff = time.time() - now
        # print(timeDiff)

    now = time.time()

    x, y, w, h = 0, 0, 175, 75

    # Draw black background rectangle
    cv2.rectangle(frame, (x, x), (x + w + 75, y + h), (0, 0, 0), -1)

    cv2.putText(frame, class_label + ': ' + str(confidence_score), (x + int(w / 10), y + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    height, width, layers = frame.shape
    new_h = height / 2
    new_w = width / 2
    resize = cv2.resize(frame, (int(new_w), int(new_h)))

    # print((1.0 - timeDiff)/fps)
    # print(time.time() - now)
    time.sleep((1.0 - timeDiff)/fps - (time.time() - now))
    # Display the resulting frame
    cv2.imshow('video', resize)

    # timeDiff = time.time() - now
    # if (timeDiff < 1.0/(fps)):
    #     print(1.0/(fps) - timeDiff)
    #     # time.sleep(1.0/(fps) - timeDiff)

    # creating 'q' as the quit
    # button for the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if flag == True:
        wave_obj = sa.WaveObject.from_wave_file("demo_audio.wav")
        wave_obj.play()
        start_time = time.time()
        flag = False

# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()



