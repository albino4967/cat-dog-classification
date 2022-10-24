import os
import numpy as np
from skimage import io
from skimage.transform import resize
import argparse
import pandas as pd
import pickle
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import imghdr

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

def model_generation(num_layers, dropout, learning_rate, momentum) :
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    for i in range(num_layers) :
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum),
                  metrics=['accuracy'])

    return model

def main(num_layers, dropout, learning_rate, momentum, epoch):
    data_dir = 'cat-dog-dataset/train_data'
    images, labels = data_generation(data_dir)

    model = model_generation(num_layers, dropout, learning_rate, momentum)
    model.fit(images, labels, epochs=epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=float, default =5, help='input int')
    parser.add_argument('--epoch', type=float, default =10, help='input int')
    parser.add_argument('--dropout', type=float, default =0.75, help='input 0~1 float')
    parser.add_argument('--learning_rate', type=float, default =0.01, help='recommended 0.01')
    parser.add_argument('--momentum', type=float, default =0.9, help='recommended 0.9')
    args = parser.parse_args()

    num_layers, epoch, dropout, learning_rate, momentum = args.num_layers, args.epoch, args.dropout, args.learning_rate, args.momentum
    main(int(num_layers), dropout, learning_rate, momentum, int(epoch))