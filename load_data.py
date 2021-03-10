from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os
from imutils import paths
import numpy as np

def mask_detect(img_shape):
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images("data"))
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]

        # load the input image and preprocess it
        image = load_img(imagePath, target_size=img_shape)
        image = img_to_array(image)
        image = preprocess_input(image)

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # 80% train and 20% test
    (x_train, x_test, y_train, y_test) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)
    
    print("[INFO] done load...")
    return x_train, x_test, y_train, y_test