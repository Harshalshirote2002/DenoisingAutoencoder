import idx2numpy
import cv2
import numpy as np
import os

filePath = 'archive/train-images-idx3-ubyte/train-images-idx3-ubyte'

folder_path = 'data/images'

arr = idx2numpy.convert_from_file(filePath)

os.makedirs(folder_path, exist_ok = True)

for i, item in enumerate(arr):
    item = np.array(item)
    image_path = os.path.join(folder_path, f'{i}.png')
    cv2.imwrite(image_path, item)

print(f"Data Extracted to {folder_path}")