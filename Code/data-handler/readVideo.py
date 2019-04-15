import cv2
import numpy as np
import os

# set video file path of input video with name and extension

#for frame identity
index = 0
for video_path in os.path.listdir(data_path): 

  vid = cv2.VideoCapture(video_path)


  if not os.path.exists('images'):
    os.makedirs('images')
  
  while(True):
    # Extract images
    ret, frame = vid.read()
    # end of frames
    if not ret: 
        break
    # Saves images
    name = './images/'+video_path+'Frame' + str(index) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # next frame
    index += 1