import cv2
import dlib
import numpy as np
import sys

print('cv2 version:', cv2.__version__)
print('dlib version:',dlib.__version__)
print('GPU compiled dlib highly recommended...')
print('hit ENTER to quit')
print('hit s to toggle capturing face for encoding')
print('... launching video window.... please wait...')


# you need a dlib compiled with GPU to work well

shape_file = 'shape_predictor_68_face_landmarks.dat'
feature_detector = dlib.shape_predictor(shape_file)

face_file = 'mmod_human_face_detector.dat'
face_detector = dlib.cnn_face_detection_model_v1(face_file)
is_hog_mode=False
if len(sys.argv) > 2:
    if sys.argv[2].lower().startswith('hog'):
        face_detector = dlib.get_frontal_face_detector()
        is_hog_mode=True
        print('is hog mode')

encode_file = 'dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(encode_file)


video_capture = cv2.VideoCapture(0)

# set how many frames to skip
step = 0
show_at_step=30

data_list = []

is_capture = False

while True:
    if step%show_at_step != 0:
        step += 1
        continue
    else:
        step = 1

    ret, frame = video_capture.read()
    
    # convert to rgb for dlib
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    dets = face_detector(img, 1)

    if is_capture:
        if is_hog_mode:
            features = [feature_detector( img , d ) for i, d in enumerate(dets)]
        else:
            features = [feature_detector( img , d.rect ) for i, d in enumerate(dets)]
        rgb_img = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        r = [np.array(face_encoder.compute_face_descriptor(rgb_img, f, 30)) for f in features][0]
        data_list.append(r)
        print('captured count:',len(data_list))

    for i, d in enumerate(dets):
        if is_hog_mode:
            x=d.left()
            y=d.top()
            x2=d.right()
            y2=d.bottom()
        else:
            x = d.rect.left()
            y = d.rect.top()
            x2 = d.rect.right()
            y2 = d.rect.bottom()
        
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
         
    cv2.imshow('Video', frame)

    key_value = cv2.waitKey(1)

    if key_value == 13 or key_value == 27 or key_value == 113: # return
        break

    elif key_value == 115: # s
        is_capture = not is_capture

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

if len(data_list) > 0:
    avg = np.mean(data_list , axis = 0)
    file_name = 'encoded' if len(sys.argv) < 2 else sys.argv[1]
    np.save('./encoding/'+file_name , avg)
    print('saved encoding for:', file_name)
