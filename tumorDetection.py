import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_set = "Data"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data = tf.keras.utils.image_dataset_from_directory(data_set)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


print("Batch shape:", batch[0].shape) 
print("Labels:", batch[1])  

# Save the first image to disk
cv2.imwrite("test_image.png", batch[0][0])
test = cv2.imread("test_image.png")
converted = cv2.cvtColor(test, cv2.COLOR_RGB2BGR)
cv2.imshow('test_image.png', converted)
cv2.waitKey(0)
cv2.destroyAllWindows()
