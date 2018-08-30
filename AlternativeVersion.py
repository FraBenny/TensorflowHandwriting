import sys
import numpy as np
from os import getcwd
from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from mnist import MNIST
from keras.utils import np_utils


def main(train):
    # Load all data
    x_train, y_train, x_test, y_test, mapping = load_data('emnist')
    model = None

    if train:
        # Train the Convolutional Neural Network
        build_model()
        train_model(model, x_train, y_train, epochs=10)

        # Save the model to 'emnist-cnn.h5'. It can be loaded afterwards with cr.load_model().
        save_model()
    else:
        # Load a trained model instead of training a new one.
        try:
            load_model()
        except:
            print('[Error] No trained CNN model found.')

    # We can use some keras' Sequential model methods too, like summary():
    model.summary()

    # Use the CNN model to recognize the characters in the test set.
    preds = read_text(x_test, mapping)
    print(preds)

    # Evaluate how well the CNN model is doing.
    # If it's not good enough, we can try training it with a higher number of epochs.
    evaluate_model(x_train, y_train)

def load_data(path, ):
    # Read all EMNIST test and train data
    mndata = MNIST(path)

    x_train, y_train = mndata.load(path + '/emnist-byclass-train-images-idx3-ubyte',
                                   path + '/emnist-byclass-train-labels-idx1-ubyte')
    x_test, y_test = mndata.load(path + '/emnist-byclass-test-images-idx3-ubyte',
                                 path + '/emnist-byclass-test-labels-idx1-ubyte')

    # Read mapping of the labels and convert ASCII values to chars
    mapping = []

    with open(path + '/emnist-byclass-mapping.txt') as f:
        for line in f:
            mapping.append(chr(int(line.split()[1])))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = normalize(x_train)
    x_test = normalize(x_test)

    x_train = reshape_for_cnn(x_train)
    x_test = reshape_for_cnn(x_test)

    y_train = preprocess_labels(y_train, len(mapping))
    y_test = preprocess_labels(y_test, len(mapping))

    return x_train, y_train, x_test, y_test, mapping


def normalize(array):
    """Normalize an array with data in an interval of [0, 255] to [0, 1].

    Args:
        array (numpy.ndarray): Data array to be normalized.
    Returns:
        Array with all values inside the interval [0, 1].
    """
    array = array.astype('float32')
    array /= 255

    return array


def reshape_for_cnn(array, color_channels=1, img_width=28, img_height=28):
    """Reshape the image data to be used in a Convolutional Neural Network.
    Args:
        array (numpy.ndarray): Data to be reshaped.
    Returns:
        Reshaped array containing all original data.
    """
    return  array.reshape(array.shape[0], color_channels, img_width, img_height)

def preprocess_labels(array, nb_classes):
    """Perform an "one-hot encoding" of a label array (multiclass).
    Args:
        array (numpy.ndarray): Array of labels (multiclass).
        nb_classes: Total number of classes.
    Returns:
        One-hot encoded label array.
    """
    return np_utils.to_categorical(array, nb_classes)


def train_model( model, X, y, batch_size=128, epochs=10):
    """Train a Convolutional Neural Network to recognize handwritten characters.

        Args:
            X (numpy.ndarray): Training data (EMNIST ByClass dataset)
            y (numpy.ndarray): Labels of the training data.
            batch_size (int): How many images the CNN should use at a time.
            epochs (int): How many times the data should be used to train the model.
    """
    callbacks.Callback()
    checkpoint = callbacks.ModelCheckpoint(filepath=getcwd()+'weights.{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.load_weights(filepath=getcwd()+'weights.04.hdf5', by_name=False)
    model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
    print(checkpoint)


def build_model(nb_classes=62, nb_filters=32, kernel_size=(5, 5), pool_size=(2, 2), input_shape=(1, 28, 28)):
    """Build a Convolutional Neural Network model to recognize handwritten characters in images.
    Args:
        nb_classes (int): The number of classes in the EMNIST dataset.
        nb_filters (int): Number of convolutional filters to be used.
        kernel_size (tuple(int, int)):  Size of the kernel (group of weights shared over the image values).
        pool_size (tuple(int, int)): Downscale factor for the MaxPooling2D layer.
        input_shape (tuple(int, int, int)): Shape of the images as (# of color channels, width, height).
    """
    model = Sequential()
    model.add(Convolution2D(int(nb_filters / 2), kernel_size, padding='valid',
                                 input_shape=input_shape, activation='relu',
                                 kernel_initializer='he_normal', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Convolution2D(nb_filters, kernel_size, activation='relu',
                                 kernel_initializer='he_normal', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(250, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(125, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='softmax', kernel_initializer='he_normal'))
    model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])


def save_model(self):
    """Save the trained model to a file."""
    self.model.save('emnist-cnn.h5', overwrite=True)


def load_model(self):
    """Load a trained model from a file."""
    self.model = load_model('emnist-cnn.h5')


def read_text(self, data, mapping):
    """Identify handwritten characters in images.
        Args:
            data (numpy.ndarray): An array containing the data of the images to be recognized.
            mapping (dict): Label mapping to convert from class to character.
        Returns:
            text (str): Text predicted from the handwritten characters.
    """
    preds = self.model.predict(data)
    preds = np.argmax(preds, axis=1)
    return ''.join(mapping[x] for x in preds)

def evaluate_model(self, X, y):
    """Evaluate the loss and accuracy of the trained model.
        Args:
            X (numpy.ndarray): Test data.
            y (numpy.ndarray): Labels of the test data.
    """
    score = self.model.evaluate(X, y)
    print('Loss:', score[0])
    print('Accuracy:', score[1])


if __name__ == '__main__':
    main(train=True)

