import cv2
import os

person_name = input("Enter person's name: ").strip()
if person_name == "":
    raise ValueError("Person name cannot be empty")

base_path = "dataset"
if not os.path.exists(base_path):
    os.makedirs(base_path)  # ensure base dataset folder exists

dataset_path = os.path.join(base_path, person_name)  # path for this person

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)   #create a folder

cap=cv2.VideoCapture(0) #start webcam capture

# Initialize count based on existing images in folder (robust way)
existing_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]  # get all .jpg files

if len(existing_files) > 0:  # if folder has images
    numbers = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]  # extract numbers
    count = max(numbers) + 1 if len(numbers) > 0 else 0  # continue from max index
else:
    count = 0  # start from 0 if empty

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
