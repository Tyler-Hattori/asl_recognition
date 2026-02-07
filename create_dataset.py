import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import skimage.measure
import cv2
import os

def create_dataset(threshold=135,type='train'):
    G = 10
    images = []
    for set in range(3):
        path = './gestures'+str(set+1)+'/'+type+'/'

        for gesture in range(G):
            gesture_idx = str(gesture)+'/'
            if set == 1: gesture_idx = 'A'+gesture_idx
            train_imgs = os.listdir(path+gesture_idx)

            for i in range(len(train_imgs)):
                img = mpimg.imread(path+gesture_idx+train_imgs[i])
                if set == 1: img = cv2.cvtColor(img[:,::-1], cv2.COLOR_BGR2HSV)[:,:,1]
                elif set == 2: img = img[:,::-1,0]

                if set == 0: threshold = 135
                else: threshold = 50

                if set == 2: px = 7
                else: px = 1

                img = np.array(binarize(img,threshold),float)
                img = trim(img,px)
                img = np.array(pool(img),int)
                seq = image2seq(img)
                entry = np.append(seq,gesture)
                images.append(entry)
    images = np.array(images,int)
    return images

def show_image(plot_images=False,set=2):
    train_path = './gestures'+str(set)+'/train/'

    gesture_idx = str(random.randint(0,9))+'/'
    if set == 2: gesture_idx = 'A'+gesture_idx
    train_imgs = os.listdir(train_path+gesture_idx)

    img_idx = random.randint(0,len(train_imgs))
    image = mpimg.imread(train_path+gesture_idx+train_imgs[img_idx])

    if set == 2: image2 = cv2.cvtColor(image[:,::-1], cv2.COLOR_BGR2HSV)[:,:,1]
    elif set == 3: image2 = image[:,::-1,0]
    else: image2 = image

    if set == 1: threshold = 135
    else: threshold = 50

    if set == 3: px = 7
    else: px = 1

    image3 = np.array(binarize(image2,threshold),float)
    image4 = trim(image3,px)
    image5 = np.array(pool(image4),int)

    images = [image,image2,image3,image4,image5]

    if plot_images:
        fig, axs = plt.subplot_mosaic([[i for i in range(len(images))]], layout='constrained')
        for label, ax in axs.items():
            ax.imshow(images[label])
        plt.show()
    return image

def trim(image,px=3):
    kernel = np.ones((px,px),np.uint8)
    return cv2.erode(image,kernel)

def pool(image):
    return cv2.resize(image,(20,20))

def binarize(image,threshold=135):
    N = np.size(image,1)
    return np.array([[1 if image[i][j] > threshold else 0 for j in range(N)] for i in range(N)],int)

def image2seq(image,Q=10):
    b = [pow(2,i) for i in range(Q)]
    image_flat = image.flatten()
    image_flat = np.append(image_flat,np.zeros(np.mod(len(image_flat),Q),int))
    seq = np.zeros(int(len(image_flat)/Q),int)
    for i in range(len(seq)):
        seq[i] = image_flat[i*Q:i*Q+Q] @ b
    return seq

if __name__=="__main__": 
    # show random image
    #show_image(True,1)

    # create dataset
    training = create_dataset(type='train')
    testing  = create_dataset(type='test')

    # split datasets into inputs and outputs
    train_dataset_ordered = np.array([training[i][:-1] for i in range(len(training))], int)
    train_labels_ordered = np.array([training[i][-1] for i in range(len(training))], int)
    test_dataset_ordered = np.array([testing[i][:-1] for i in range(len(testing))], int)
    test_labels_ordered = np.array([testing[i][-1] for i in range(len(testing))], int)

    # shuffle dataset entries
    np.random.shuffle(training)
    np.random.shuffle(testing)

    # split datasets into inputs and outputs
    train_dataset = np.array([training[i][:-1] for i in range(len(training))], int)
    train_labels = np.array([training[i][-1] for i in range(len(training))], int)
    test_dataset = np.array([testing[i][:-1] for i in range(len(testing))], int)
    test_labels = np.array([testing[i][-1] for i in range(len(testing))], int)

    np.save('./dataset/training_sequences',train_dataset)
    np.save('./dataset/training_labels',train_labels)
    np.save('./dataset/testing_sequences',test_dataset)
    np.save('./dataset/testing_labels',test_labels)

    np.save('./dataset/testing_sequences_ordered',test_dataset_ordered)
    np.save('./dataset/testing_labels_ordered',test_labels_ordered)