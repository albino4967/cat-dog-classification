import os
import numpy as np
from skimage import io
from skimage.transform import resize
import argparse
import pandas as pd
import pickle
from tensorflow.keras import datasets, layers, models, applications
import tensorflow as tf
import imghdr
import matplotlib.pyplot as plt

def plot_image_from_prediction(img, prediction) :
    plt.imshow(img)
    plt.savefig(f'{prediction}.jpg')

def data_generation(pre_data):
    file_list = os.listdir(pre_data)
    train_list = []
    label_list = []
    for i, item in enumerate(file_list):
        temp = f'{pre_data}/{item}'
        if imghdr.what(temp) == 'jpeg' :
            img = io.imread(temp)
            if img is not None :
                resize_img = resize(img,(224,224))
                resize_img = np.array(resize_img)
                train_list.append(resize_img)
                if 'cat' in item :
                    label_list.append([1,0])
                elif 'dog' in item :
                    label_list.append([0,1])

    train_list = np.array(train_list)
    label_list = np.array(label_list)
    return train_list, label_list



def model_prediction(model, test_path) :
    file_list = os.listdir(test_path)
    test_list = []

    class_name = ['cat', 'dog']

    for i, item in enumerate(file_list):
        temp = f'{test_path}/{item}'
        if imghdr.what(temp) == 'jpeg' :
            img = io.imread(temp)
            if img is not None :
                resize_img = resize(img,(224,224))
                resize_img = np.array(resize_img)
                test_list.append(resize_img)

    test_image = tf.expand_dims(test_list[11], 0)
    prediction = model.predict(test_image)
    results = tf.math.argmax(tf.nn.softmax(prediction[0]))
    plot_image_from_prediction(test_list[10], class_name[results])

def model_generation() :
    model = models.Sequential()

    model.add(applications.ResNet50(include_top=True, weights=None, input_shape=(224, 224, 3), classes=2))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def main(epoch):
    train_data_dir = 'cat-dog-dataset/train_data'
    test_data_dir = 'cat-dog-dataset/test_data'
    images, labels = data_generation(train_data_dir)

    model = model_generation()
    model.fit(images, labels, epochs=epoch)
    model_prediction(model, test_data_dir)
    model.save('classification_model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=float, default =1, help='input int')
    args = parser.parse_args()
    epoch = args.epoch

    main(int(epoch))