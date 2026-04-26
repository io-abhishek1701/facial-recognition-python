import os

os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tensorflow.keras.models import Sequential       #CNN Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense        #layers

def create_model(num_classes):
    model = Sequential()        #create an empty model

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)))  
    # First layer to accept input image

    model.add(MaxPooling2D((2,2)))  

    model.add(Conv2D(64, (3,3), activation='relu'))  

    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation='relu'))

    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())  

    model.add(Dense(128, activation='relu'))  

    model.add(Dense(num_classes, activation='softmax'))  

    model.compile(
        optimizer='adam',  
        loss='sparse_categorical_crossentropy',  
        metrics=['accuracy']
    )
    print("Model Training complete")
    return model

if __name__ == "__main__":
    print("This file defines the CNN model. Run train.py instead.")