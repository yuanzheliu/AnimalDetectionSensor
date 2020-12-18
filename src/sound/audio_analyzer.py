import pyaudio
import numpy as np
from tkinter import *

class DummyPredict():
    def __init__(self):
        pass
    def predict(self, input):
        return ("human", sum(input) % 100)

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
        out = "Predicted {}, {}% confident".format(predict[0], predict[1])
        self.label.configure(text = out)
        self.after(2000, self.update_clock)

canvas = Tk()
demo=DemoClassify(canvas)
canvas.wm_title("Tkinter clock")
canvas.geometry("450x200")
canvas.after(1, demo.update_clock)
canvas.mainloop()

