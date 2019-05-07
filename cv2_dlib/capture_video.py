import cv2

cap = cv2.VideoCapture(0)
while True:
    # read video image
    (_,image)=cap.read()
    
    #convert to gray and show image in a window
    out_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('capture window', out_image)
    
    # capture key and if return key exit
    key = cv2.waitkey(1)
    if key == 13:
        break

cap.release()
cv2.destoryWindow('capture window')