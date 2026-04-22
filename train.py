import numpy as np
from data_converter import data,labels,people
from cnn_model import create_model
import os  # Import os module

os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"  # Limit TensorFlow internal threads
os.environ["TF_NUM_INTEROP_THREADS"] = "1"  # Limit TensorFlow parallel threads
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix Mac threading conflicts

import tensorflow as tf  # Import TensorFlow AFTER setting env variables

tf.config.threading.set_intra_op_parallelism_threads(1)  # Limit intra threads
tf.config.threading.set_inter_op_parallelism_threads(1)  # Limit inter threads

model = create_model(len(people))

model.fit(data,labels,epochs = 10,batch_size = 8)

model.save("face_model.h5")

print("Training Completed")