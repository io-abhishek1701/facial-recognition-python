import numpy as np
import os
import cv2

data = []
labels = []

dataset_path = "dataset"

people = [p for p in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, p))]

for label, person in enumerate(people):
    person_path = os.path.join(dataset_path,person)

    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            data.append(img)
            labels.append(label)

data = np.array(data)
labels = np.array(labels)

data = data / 255.0     #Normalize i.e for 0-1
data=data.reshape(-1,128,128,1)

print(data)