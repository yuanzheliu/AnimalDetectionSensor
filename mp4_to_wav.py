from moviepy.editor import *
video = VideoFileClip("demo_video.mp4")
video.audio.write_audiofile("demo_audio.wav")