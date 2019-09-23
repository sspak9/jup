# ### A better Face Detection using Dlib
# Dlib site: http://dlib.net/
# 
# Qutoe from the site:
# 
# *Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems.*
# 
# For any image processing, CUDA (i.e. NVidia GPU processing library ) would be leveraged when compiling.
# 
# This demo is using a CPU, so the "refresh" of the video stream will be about once every 60 seconds or so
# 
# You will see immediate improvement in detecting faces even when the face slanted or viewed from the side.
# 
# The facial feature detection attempts to identify where the eye brow, yes, nose, mouth, and chin are
# 

import cv2
import dlib
import numpy as np
import glob
import os
import sys

print('cv2 version:', cv2.__version__)
print('dlib version:',dlib.__version__)
print('GPU compiled dlib highly recommended...')
print('hit ENTER to quit')
print('hit s to toggle drawing square around the face')
print('hit f to toggle feature detection when face detection is enabled')
print('hit b to toggle applying blur to the face when face detection is enabled')
print('... launching video window.... please wait...')

# you need a dlib compiled with GPU to work well

shape_file = 'shape_predictor_68_face_landmarks.dat'
feature_detector = dlib.shape_predictor(shape_file)

face_file = 'mmod_human_face_detector.dat'
face_detector = dlib.cnn_face_detection_model_v1(face_file)

if len(sys.argv) > 1:
  if sys.argv[1].lower().startswith('hong'):
    face_detector = dlib.get_frontal_face_detector()


encode_file = 'dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(encode_file)

#sang_encoding = np.load('sang2.npy')
npy_list = glob.glob('./encoding/*.npy')
enc_dict = {}
for file_name in npy_list:
    e = np.load(file_name)
    basename = os.path.basename(file_name)
    name , _ = os.path.splitext(basename)
    enc_dict[name] = e

is_detect_feature = True
is_detect_face = True
is_blur = False

video_capture = cv2.VideoCapture(0)

# set how many frames to skip
step = 0
show_at_step=2

while True:
    if step%show_at_step != 0:
        step += 1
        continue
    else:
        step = 1

    ret, frame = video_capture.read()
    
    if is_detect_face:
        # convert to rgb for dlib
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detect faces
        dets = face_detector(img, 1)
        
        # rectangles around faces
        for i, d in enumerate(dets):
            x = d.rect.left()
            y = d.rect.top()
            x2 = d.rect.right()
            y2 = d.rect.bottom()
            
            if not is_blur:
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 1)
            else:
                #blur the frame
                top = int(y*0.80)
                bottom = int(y2*1.1)
                left = int(x*0.9)
                right = int(x2*1.1)
                
                face_image = frame[top:bottom, left:right]

                # Blur the face image
                face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

                # Put the blurred face region back into the frame image
                frame[top:bottom, left:right] = face_image
            
            if not is_blur:
                #get facial features
                coor = feature_detector(img , d.rect)

                # must have detected the features to go foward
                if not coor is None:
                    if is_detect_feature:
                        for f in range(0,68):
                            cv2.circle(frame , (coor.part(f).x , coor.part(f).y) , 1, (255,255,255),1 )

                    # find matching face using encoding
                    rgb_img = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB) # i think we have to use RGB image
                    encoding = np.array(face_encoder.compute_face_descriptor(rgb_img, coor, 30))
                
                    # find first match. TODO change this to use single np.linalg.norm (list - single , axis = 1)
                    dist_dict={}
                    dist_list=[]
                    for key in enc_dict:
                        enc = enc_dict[key]
                        dist = np.linalg.norm(enc - encoding)
                        if dist < 0.6:
                            dist_dict[dist] = key
                            dist_list.append(dist)
                    
                    if len(dist_list) > 0:
                        dist_list.sort()
                        name = dist_dict[dist_list[0]]
                        cv2.putText( frame , name , (x,y-10) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                        
                
    cv2.imshow('Video', frame)

    key_value = cv2.waitKey(1)
    if key_value == 13 or key_value == 27 or key_value == 113: # return
        break
    elif key_value == 102: # f
        is_detect_feature = not is_detect_feature

    elif key_value == 115: # s
        is_detect_face = not is_detect_face

    elif key_value == 98: #b
        is_blur = not is_blur

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

