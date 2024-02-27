import cv2
import numpy as np
from os import listdir
from os.path import isfile,join
import time

data_path = 'C:/Users/Admin/Downloads/Final year project/CCTV THEIF DETECTION/face recognition/faces/'
onlyfiles =[f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels =[],[]

for i,files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)
model=cv2.face.createFisherFaceRecognizer()


model.train(np.asarray(Training_Data),np.asarray(Labels))
print("Model Training is complete !!!")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_detector(img,size=0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 2)

    if faces is():
        return img,[]
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h,x:x+w]
        roi = cv2.resize(roi,(200,200))
    return img,roi
cap =cv2.VideoCapture(0)
while True:
    ret, frame =cap.read()

    image, face =face_detector(frame)

    try:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1]<500:
            confidence =int(100*(1-(result[1])/300))
            display_string = str(confidence)+'Confidence it is user'
        cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

        if confidence > 80:

            cv2.putText(image,"Thief Found",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow("face cropper",image)
          
        elif(confidence>65 or confidence <80):
            print(confidence)
            cv2.putText(image," ",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow("face cropper",image)
        else:
          print()
          
            
            
    except:
        
        #cv2.putText(image,"face Not found",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.imshow("face cropper",image)


    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()


















    
