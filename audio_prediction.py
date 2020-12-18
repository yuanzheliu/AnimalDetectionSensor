import pyaudio
import numpy as np
from tkinter import *
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf

class DummyPredict():
    def __init__(self):
        self.num_classes = 13
        chose_label = ['dog','chirping_birds','thunderstorm','sheep','water_drops',
        'wind','footsteps','frog','cow','rain','insects','breathing','cat']

        class_dict = {i:x for x,i in enumerate(chose_label)}
        num_dict = {}
        for key, value in class_dict.items():
            num_dict[value] = key
        self.INPUTSHAPE = (13,87,1)
        multi_model = self.create_model()
        binary_model = self.create_binary_model()
        ## restore the weights for multi_class_model
        multi_latest = tf.train.latest_checkpoint('cpkt')
        multi_model.load_weights(multi_latest)
        ## restore the weights for binary_class_Model
        bi_latest = tf.train.latest_checkpoint('binary_cpkt')
        binary_model.load_weights(bi_latest)

        ## assign self
        self.multi_model = multi_model
        self.binary_model = binary_model
        self.num_dict = num_dict

    def create_model(self):
        model =  tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16 , (3,3),activation = 'relu',padding='valid', input_shape = self.INPUTSHAPE),
            tf.keras.layers.Conv2D(16, (3,3), activation='relu',padding='valid'),

            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='valid'),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='valid'),

            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='valid'),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='valid'),
            tf.keras.layers.GlobalAveragePooling2D(),
            
            tf.keras.layers.Dense(32 , activation = 'relu'),
            tf.keras.layers.Dense(self.num_classes , activation = 'softmax')
        ])
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        return model
    
    def create_binary_model(self):
        model =  tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16 , (3,3),activation = 'relu',padding='valid', input_shape = self.INPUTSHAPE),
            tf.keras.layers.Conv2D(16, (3,3), activation='relu',padding='valid'),

            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='valid'),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='valid'),

            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='valid'),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='valid'),
            tf.keras.layers.GlobalAveragePooling2D(),
            
            tf.keras.layers.Dense(32 , activation = 'relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
        ])
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        return model

        
    def predict(self, input):
        mfcc_ = librosa.feature.mfcc(input , sr=22050, n_mfcc=13)
        x = [mfcc_]
        x = np.array(x)

        # multi_class_prediction
        predicted_class = self.multi_model.predict_classes(x)
        class_name = self.num_dict[predicted_class[0]]

        # binary_class_prediction
        predicted_bi = self.binary_model.predict_classes(x)
        predicted_bi = predicted_bi.flatten()
        bi_name = 'Non-Animal'
        if predicted_bi[0] == 1:
            bi_name = 'Animal'

        ## get the confidence
        multi_conf = np.max(self.multi_model.predict(x)[0]) 
        bi_conf = np.max(self.binary_model.predict(x)[0])

        return class_name, multi_conf*100, bi_name, bi_conf*100


class DemoClassify(Frame):
    def __init__(self, master = None):
        # -------------------GUI initialization--------------------------
        Frame.__init__(self,master)
        self.master = master
        self.label = Label(text="", font=("Helvetica", 18))
        self.label.place(x=50,y=80)
        self.model = DummyPredict()
        # ----------------Audio input initization-------------------------
        RATE = 22050 # sampling rate
        self.CHUNK = 2*RATE # number of data to read every time
        p=pyaudio.PyAudio() 
        self.stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                    frames_per_buffer=self.CHUNK) #uses default input device

        # --------------------Start loop-------------------------------
        self.update_clock()

    def read_audio(self): 
        return  np.frombuffer(self.stream.read(self.CHUNK),dtype=np.int16)

    def update_clock(self):
        input = self.read_audio()
        print(input.shape)
        predict = self.model.predict(input)
        out = "Multi-class Predicted {} with {}% confidence \n Animal Vs. Nonanimal Predicted {} with {}% confidence".format(predict[0], predict[1], predict[2], predict[3])
        self.label.configure(text = out)
        self.after(2000, self.update_clock)

canvas = Tk()
demo=DemoClassify(canvas)
canvas.wm_title("Tkinter clock")
canvas.geometry("450x250")
canvas.after(1000, demo.update_clock)
canvas.mainloop()

