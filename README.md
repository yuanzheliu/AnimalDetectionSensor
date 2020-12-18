# AnimalDetectionSensor
Course Project for Machine Learning and Sensing

#  Usage
## Audio Analyzer
### Require pyaudio to get audio stream to python (tested on Ubuntu)
```
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
pip3 install pyaudio
```
###  Require tkinter to draw GUI

```
sudo apt-get install python3-tk
```

### (optional) To get best result, redirect audio output to audio input instead collect with microphone

**macOS**

Use [Soundflower](https://github.com/mattingalls/Soundflower/releases/) to create a virtual audio device. You can follow this [tutorial](https://apple.stackexchange.com/questions/221980/os-x-route-audio-output-to-audio-input).

**Ubuntu**

Follow this [tutorial](https://unix.stackexchange.com/questions/82259/how-to-pipe-audio-output-to-mic-input).


