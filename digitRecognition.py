import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="Train a model on MNIST data",  action='store_true')
parser.add_argument("-r", "--recognise", help="Enter image filename and predict digit")
args = parser.parse_args()

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

import matplotlib.pyplot as plt

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras.preprocessing import image

# save / load model
from keras.models import load_model

# convolutional neural network (cnn)
from keras import backend as K
K.set_image_dim_ordering('th')

if args.recognise:
    print("Selected Recognise")

    # get filemname argument
    filename = args.recognise

    print("The file you chose is: " + filename)

    # convert to grayscale and 28x28
    img = image.load_img(path=filename,color_mode = "grayscale",target_size=(28,28,1))
    img = image.img_to_array(img)

    test_img = img.reshape((1,784))

    # add dimension to match with expected array
    test_img = np.expand_dims(img, axis=0)

    # load model from h5 file
    model = keras.models.load_model('model.h5')

    # predict the digit
    img_class = model.predict_classes(test_img)
    # first index as there is only one image predicted
    prediction = img_class[0]

    # classname = img_class[0]

    print("Class: ",prediction)

    # display the processed image
    img = img.reshape((28,28))
    plt.imshow(img)
    plt.title(prediction)
    plt.show()

# option -t
if args.train:
    print("Selected train")
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode output variables
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    def base_model():
        # create model
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # compile
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # train model
    # build model
    model = base_model()

    # Fit model
    # epoch = iteration over all data, batch size = no. of images at a time
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

    # save model
    model.save('model.h5')

    # eval
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))
    # train_model()
    # build model
    model = base_model()

    # Fit model
    # epoch = iteration over all data, batch size = no. of images at a time
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

    # save model
    model.save('model.h5')

    # eval
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))



    

# def train_model():
#     # build model
#     model = base_model()

#     # Fit model
#     # epoch = iteration over all data, batch size = no. of images at a time
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

#     # save model
#     model.save('model.h5')

#     # eval
#     scores = model.evaluate(X_test, y_test, verbose=0)
#     print("CNN Error: %.2f%%" % (100-scores[1]*100))