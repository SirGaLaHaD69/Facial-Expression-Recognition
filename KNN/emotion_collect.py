import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

face_data=[]
file_name=input("Enter Emotion")
dataset_path = './emotions/'
skip=0
while True:
    ret,frame = cap.read()

    if ret == False:
        continue
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    if len(faces)==0:
        continue

    #Picking the largest face acc to Area( w*h  i.e  f[2]*f[3] where  'f' is the face tuple)


    for (x,y,w,h,) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(250,255,100),2)

        #Extract (Crop out the required Area) : region Of interest

        offset =10
        face_section = frame[y-offset:y+offset+h,x-offset:x+offset+w]
        face_section = cv2.resize(face_section,(100,100))
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow('Video Frame',frame)
    cv2.imshow('Face Section',face_section)



    #Wait for User input

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# convert our face list array into a numpy array
face_data = np.asarray(face_data,)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save into file .npy
np.save(dataset_path+file_name+'.npy',face_data)
print('Data Saved Succesfully at'+dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()
