# Copyright 2021 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Two step v2-compatible pipeline."""
import os
import random
import kfp
from kfp import components, dsl
from kfp.components import InputPath, OutputPath
from kfp.components import func_to_container_op
from typing import NamedTuple
import requests

def download_file_from_google_drive(
        output_dataset_zipfile: OutputPath('Dataset')
):
    import gdown
    import zipfile
    import os
    zip_file_name = "cat-dog-dataset.zip"
    id_ = '1Z9JUrCDGaAJBGfmCw2BjM0qY-G9xLhLm'
    url = f'https://drive.google.com/uc?id={id_}'
    os.mkdir(output_dataset_zipfile)
    gdown.download(url, f'{output_dataset_zipfile}/{zip_file_name}', quiet=False)
    print(f'{output_dataset_zipfile}/{zip_file_name} download complete!')

    print(os.path.isdir(output_dataset_zipfile))
    print(os.path.isfile(f'{output_dataset_zipfile}/{zip_file_name}'))
    with zipfile.ZipFile(f'{output_dataset_zipfile}/{zip_file_name}', 'r') as existing_zip:
        existing_zip.extractall(output_dataset_zipfile)

    print(os.listdir(output_dataset_zipfile))

download_file_from_google_drive_op = components.create_component_from_func(
    download_file_from_google_drive, base_image='pytorch/pytorch',
    #output_component_file = 'train_data.pickle',
    packages_to_install=['gdown']
)

def data_generation(
        pre_data: InputPath('Dataset'),
        train_data: OutputPath('Dataset')
):
    from skimage import io
    from skimage.transform import resize
    import numpy as np
    import os
    import pandas as pd
    import pickle

    train_data_dir = f'{pre_data}/train_data'
    file_list = os.listdir(train_data_dir)
    train_list = []
    label_list = []
    for i, item in enumerate(file_list):
        temp = f'{train_data_dir}/{item}'
        img = io.imread(temp)
        if img is not None :
            resize_img = resize(img,(224,224))
            resize_img = np.array(resize_img)
            train_list.append(resize_img)
            if 'cat' in item :
                label_list.append([1,0])
            elif 'dog' in item :
                label_list.append([0,1])


    df = pd.DataFrame(columns=['image', 'label'])
    for i, image in enumerate(train_list):
        df.loc[i] = ({'image': image, 'label': label_list[i]})

    with open(train_data, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

data_generation_op = components.create_component_from_func(
    data_generation, base_image='python:3.9',
    packages_to_install=['numpy', 'scikit-image', 'pandas']
)

def checked_data_image_list(
        pre_data:InputPath('Dataset')
):
    import os
    train_data_dir = f'{pre_data}/train_data'
    file_list = os.listdir(train_data_dir)
    dog_list = []
    cat_list = []
    for i, item in enumerate(file_list):
        if 'cat' in item :
            dog_list.append(item)
        elif 'dog' in item :
            cat_list.append(item)

    print(f'train_dog_image_num : {len(dog_list)}')
    print(f'train_cat_image_num : {len(cat_list)}')

checked_data_image_list_op = components.create_component_from_func(
    checked_data_image_list, base_image='python:3.9'
)

def model_generation(
        pretrain_model : OutputPath('TFModel')
) :
    from keras.applications import ResNet50
    from keras.models import Sequential

    model = Sequential()
    model.add(ResNet50(include_top=True, weights=None, input_shape=(224, 224, 3), classes=2))
    # model = ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3), pooling=max, classes=2)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.save(pretrain_model)

model_generation_op = components.create_component_from_func(
    model_generation, base_image='tensorflow/tensorflow'
)
def train_model(
        train_dataset: InputPath('Dataset'),
        pre_model: InputPath('TFModel'),
        trained_model : OutputPath('TFModel')
) :
    import pickle
    from tensorflow import keras
    import numpy as np
    import pandas as pd

    with open(train_dataset, 'rb') as file:
        tr_data = pickle.load(file)

    images = []
    labels = []
    for i, item in enumerate(tr_data['image']) :
        images.append(item)
        labels.append(tr_data['label'][i])

    images = np.array(images)
    labels = np.array(labels)
    model = keras.models.load_model(pre_model)

    model.fit(images, labels, epochs=20)
    model.save(trained_model)

train_result_op = components.create_component_from_func(
    train_model,
    base_image='tensorflow/tensorflow',
    packages_to_install=['pandas==1.4.2']
)

@dsl.pipeline(name='example data load from s3 and train')
def aws_dog_cat_classification_pipeline():
    train_data_load_task = download_file_from_google_drive_op()
    checked_data_image_list_task = checked_data_image_list_op(train_data_load_task.outputs['output_dataset_zipfile'])
    model_generation_task = model_generation_op()
    data_generation_task = data_generation_op(train_data_load_task.outputs['output_dataset_zipfile'])
    train_task = train_result_op(
        data_generation_task.outputs['train_data'],
        model_generation_task.outputs['pretrain_model']
    )

if __name__ == '__main__':
    # Compiling the pipeline
    import kfp
    kfp.compiler.Compiler().compile(aws_dog_cat_classification_pipeline, 'cat-dog-classification.tar.gz')