import os
import time
import shutil
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torchvision
from torchvision.datasets import ImageFolder

# get_ipython().run_line_magic('matplotlib', 'inline')

# Show some information of Kaggle's input folder.

print(os.listdir("../"))
print(os.listdir("./data"))
print(os.listdir("./data/train"))
print(os.listdir("./data/test")[:6])


# In[ ]:


# ImageFolder() needs subfolders.
# Copy test images of input to temporary folder to avoid train images.

def copytree_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)
    return True

copytree_and_overwrite("./data/test", "../working/tmp/test/test_images")

print(os.listdir("../working/tmp/test/test_images")[:6])


# In[ ]:


# Read and resize images to [224, 224].

def read_image_folder(resize_shape, image_folder):
    resize = torchvision.transforms.Resize(resize_shape)
    image_folder = ImageFolder(image_folder, transform=resize)

    idx_to_class = {value: key for key, value in image_folder.class_to_idx.items()}
    image_paths = [item[0] for item in image_folder.imgs]

    image_shape = np.array(image_folder[0][0]).shape
    data_length = len(image_folder)

    data_shape = list(image_shape)
    data_shape.insert(0, data_length)

    data = np.zeros(data_shape, dtype=np.uint8)
    labels = np.zeros([data_length], dtype=np.int64)

    i = 0
    for image, label in tqdm(image_folder, desc="Reading Images"):
        data[i] = np.array(image)
        labels[i] = label
        i += 1

    data_dict = {"data": data, "labels": labels, 'data_shape': image_shape}
    info_dict = {"label_names": idx_to_class}

    return data_dict, info_dict

train_dict, train_info_dict = read_image_folder((224,224),"./data/train")
test_dict, test_info_dict = read_image_folder((224,224),"../working/tmp/test/")

np.savez("plant-train-data", **train_dict)
np.savez("plant-train-info", **train_info_dict)
np.savez("plant-test-data", **test_dict)
np.savez("plant-test-info", **test_info_dict)


