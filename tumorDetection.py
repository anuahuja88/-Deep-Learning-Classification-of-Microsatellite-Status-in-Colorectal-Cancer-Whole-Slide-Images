import tensorflow as tf
import os
import cv2

false_set = "FALSE_Group_for_Microsatellite_instability_pMMR_MSS/tiles"
true_set = "TRUE_Group_for_Microsatellite_Instability_dMMR_MSI/tiles"

for image_path in os.listdir(false_set):
    img = cv2.imread(image_path)

for image_path in os.listdir(true_set):
    img = cv2.imread(image_path)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


