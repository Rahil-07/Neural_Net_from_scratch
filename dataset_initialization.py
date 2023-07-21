import os
import cv2
import numpy as np
import random

DATADIR = './datasets/'
CATEGORIES = ["Car","Bike"]

# images_data list used to contain 'image_pixel' and 'categories'
# for Car y=0 and Bike y=1
images_data = []

def create_image():
    
    for categories in CATEGORIES:
        path = os.path.join(DATADIR,categories)
        class_num = CATEGORIES.index(categories)
        image_count = 0
        
        for img_path in os.listdir(path):
#           read total 2000 images from each car and bike datasets
            if image_count != 2000:
                try:
                    img_array_bgr = cv2.imread(os.path.join(path, img_path))
                    img_array_rgb = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)
                    img_cropped = cv2.resize(img_array_rgb, (64,64))
                    images_data.append([img_cropped,class_num])
                    image_count += 1
                
                except:
                    pass

create_image()


# split datasets into train_set and test_set
# first 1750 images in each car and bike datasets append in train_set
# remaining 250 images from each car and bike datasets append in test_set
train_set = images_data[:1750]
train_set.extend(images_data[2000:3750])

test_set = images_data[1750:2000]
test_set.extend(images_data[3750:])

random.shuffle(train_set)
random.shuffle(test_set)



train_x_set_org = []
train_y_set = []
for i in range(3500):
    train_x_set_org.append(train_set[i][0])
    train_y_set.append(train_set[i][1])

test_x_set_org = []
test_y_set = []
for i in range(500):
    test_x_set_org.append(test_set[i][0])
    test_y_set.append(test_set[i][1])

train_x_set_org = np.array(train_x_set_org)
train_y_set = np.array(train_y_set).reshape(1,3500)

test_x_set_org = np.array(test_x_set_org)
test_y_set = np.array(test_y_set).reshape(1,500)

print('Shape of Training-X set before flatten: ', train_x_set_org.shape)
print('Shape of Testing-X set before flatten: ', test_x_set_org.shape)
print('Shape of Training-y set: ', train_y_set.shape)
print('Shape of Testing-y set: ', test_y_set.shape)

train_x_set_flatten = train_x_set_org.reshape(train_x_set_org.shape[0], -1).T
test_x_set_flatten = test_x_set_org.reshape(test_x_set_org.shape[0], -1).T

train_x_set = train_x_set_flatten/255.
test_x_set = test_x_set_flatten/255.

print('Shape of Training-X set after flatten: ', train_x_set.shape)
print('Shape of Testing-X set after flatten: ', test_x_set.shape)