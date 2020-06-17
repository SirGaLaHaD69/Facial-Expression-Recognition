#importing the Libraries

import numpy as np
import cv2
import os
import trainfer

emotions={0:'Angry',
          1:'Surprised',
          2:"Sad/Calm",
          3:"Happy",
          4:"Calm/Sad"
  }



# Init Camera
cap = cv2.VideoCapture(0)
# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
trainfer.model.load_weights('best_model.h5')
while True:
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h = face

        #Extract (Crop out the required Area) : region Of interest
        offset =10
        face_section = frame[y-offset:y+offset+h,x-offset:x+offset+w]
        face_section = cv2.resize(face_section,(48,48))
        out = trainfer.model.predict(face_section.reshape(1,48,48,1))
        ind = np.argmax(out)
        pred_per = (out[0][ind])*100
        pred_name = emotions[ind]
        cv2.putText(frame,pred_name,(x,y+h+25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(frame,str(round(pred_per,2))+"%",(x+w-5,y+h+25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,33,120),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("FACES",frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()