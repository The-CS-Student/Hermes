from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    LeakyReLU,
    Add,
    BatchNormalization,
    Dense,
    GlobalAveragePooling2D,
    Dropout
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
(images,labels) = ([],[])
folder = 'asl_alphabet_train/asl_alphabet_train'

for (subdirs, dirs, files) in os.walk(folder):
    for subdir in dirs:
        subjectpath = os.path.join(folder, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            labels.append(subdir)
            images.append(np.array(Image.open(path)))
            break
print(len(images),len(labels))
train_gen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        validation_split = 0.2
)


train_dir = 'asl_alphabet_train/asl_alphabet_train'

train_data = train_gen.flow_from_directory(
                train_dir,
                target_size = (200,200),
                batch_size = 32,
                class_mode = 'categorical',
                subset = 'training'
)

validation_dir = ''
validation_data = train_gen.flow_from_directory(
                train_dir,
                target_size = (200,200),
                batch_size = 32,
                class_mode = 'categorical',
                subset = 'validation'
)

classes = list(train_data.class_indices)
def DarknetConv(inputs, filters, kernel_size, strides):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    return x

def DarknetResidual(inputs, filters):
    shortcut = inputs
    x = DarknetConv(inputs, filters//2,kernel_size=(1,1),strides=(1,1))
    x = DarknetConv(x, filters, kernel_size=(3,3),strides=(1,1))
    x = Add()([x, shortcut])
    return x
inputs = Input(shape=(200,200,3))
x = DarknetConv(inputs, 32, kernel_size=(3,3),strides=(1,1))

x = DarknetConv(x, 64, kernel_size=(3,3), strides=(2,2))

for _ in range(1):
    x = DarknetResidual(x, 64)
    
x = DarknetConv(x, 128, kernel_size=(3,3), strides=(2,2))

for _ in range(2):
    x = DarknetResidual(x, 128)
    
x = DarknetConv(x, 256, kernel_size=(3,3), strides=(2,2))

for _ in range(8):
    x = DarknetResidual(x, 256)
    
x = DarknetConv(x, 512, kernel_size=(3,3), strides=(2,2))

for _ in range(8):
    x = DarknetResidual(x, 512)
    
x = DarknetConv(x, 1024, kernel_size=(3,3), strides=(2,2))

for _ in range(4):
    x = DarknetResidual(x, 1024)
    
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)


output = Dense(29,activation='softmax')(x)

darknet = Model(inputs, output)
darknet.summary()
early_stopping = EarlyStopping(
        monitor='val_loss',patience=5,
        min_delta = 0.001
)
reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',patience=5
)
darknet.compile(optimizer='adam', loss='categorical_crossentropy',
               metrics=['acc'])

history = darknet.fit_generator(train_data,
                               epochs=50,
                               validation_data=validation_data,
                               callbacks=[early_stopping, reduce_lr])

darknet.save("weights.h5")