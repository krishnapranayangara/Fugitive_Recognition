import cv2
import numpy as np

#def nothing(x):
#    pass

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#cv2.namedWindow("Frame")
#cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)

count=0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #neighbours = cv2.getTrackbarPos("Neighbours", "Frame")

    faces = face_cascade.detectMultiScale(gray, 1.3, 2)
    if faces is():
        print("face not found !")
    else:
        count+=1
        for (x,y,h,w) in faces:
            cf=frame[y:y+h,x:x+w]
            #(x, y, w, h) = rect
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            face =cv2.resize(cf,(600,600))
            face =cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            file_name_path='C:/Users/Admin/Downloads/Final year project/CCTV THEIF DETECTION/face recognition/faces/'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            cv2.imshow("Frame", face)

    key = cv2.waitKey(1)
    if key == 27 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
