import cv2
import os

person_name=""
dataset_path="dataset/"+person_name

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)   #create a folder

cap=cv2.VideoCapture(0) #start webcam capture

count=0

box_size=600

while True:
    ret, frame = cap.read()     #Capture a frame

    if not ret:
        print("Capture Fail")
        break

    grey_conv=cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    h,w= grey_conv.shape

    cx,cy=w//2,h//2

    x1=cx - box_size//2
    y1=cy - box_size//2
    x2=cx + box_size//2
    y2=cy + box_size//2

    cv2.rectangle(grey_conv,(x1,y1),(x2,y2),(255,255,255),2)

    cv2.imshow("Capture (Align face in box)", grey_conv)      #show the processed image

    if cv2.waitKey(1) & 0xFF == ord('s'):
        face_region = grey_conv[y1:y2,x1:x2]
        resized_img = cv2.resize(face_region,(128,128))
        file_name = dataset_path + "/" + str(count) + ".jpg"
        cv2.imwrite(file_name, resized_img)
        print("Image saved as : ",file_name)
        count+=1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Capture Complete")
        break

cap.release()
cv2.destroyAllWindows()
