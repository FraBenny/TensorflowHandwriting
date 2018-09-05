import numpy as np
from os import getcwd
from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, ZeroPadding2D
import h5py


class CharRecognizer(object):
    """Handwritten character recognition with a Convolutional Neural Network."""

    def __init__(self):
        self.model = None

    def train_model(self, X, y, batch_size=1, epochs=10):
        """Train a Convolutional Neural Network to recognize handwritten characters.

        Args:
            X (numpy.ndarray)8: Training data (EMNIST ByClass dataset)
            y (numpy.ndarray): Labels of the training data.
            batch_size (int): How many images the CNN should use at a time.
            epochs (int): How many times the data should be used to train the model.
        """
        self.build_model()
        callbacks.Callback()
        checkpoint = callbacks.ModelCheckpoint(filepath=getcwd()+'weights.{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        #self.model.load_weights(filepath=getcwd()+'weights.04.hdf5', by_name=False)
        callbacks.TensorBoard(log_dir= getcwd()+'weights.{epoch:02d}')
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])

    def evaluate_model(self, X, y):
        """Evaluate the loss and accuracy of the trained model.
        Args:
            X (numpy.ndarray): Test data.
            y (numpy.ndarray): Labels of the test data.
        """
        score = self.model.evaluate(X, y)
        print('Loss:', score[0])
        print('Accuracy:', score[1])

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

    def build_model(self, nb_classes=62, nb_filters=32, kernel_size=(3, 3), pool_size=(2, 2), input_shape=(3, 28, 28)):
        """Build a Convolutional Neural Network model to recognize handwritten characters in images.
        Args:
            nb_classes (int): The number of classes in the EMNIST dataset.
            nb_filters (int): Number of convolutional filters to be used.
            kernel_size (tuple(int, int)):  Size of the kernel (group of weights shared over the image values).
            pool_size (tuple(int, int)): Downscale factor for the MaxPooling2D layer.
            input_shape (tuple(int, int, int)): Shape of the images as (# of color channels, width, height).
            """
        weights_path = 'emnist/vgg16_weights.h5'
        img_width, img_height = 28, 28

            # build the VGG16 network
        bottomModel = Sequential()
        bottomModel.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
        bottomModel.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1', data_format = 'channels_first'))
        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_2'))
        bottomModel.add(MaxPooling2D((2, 2), strides=(2, 2)))

        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_1'))
        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_2'))
        bottomModel.add(MaxPooling2D((2, 2), strides=(2, 2)))

        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_1'))
        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_2'))
        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_3'))
        bottomModel.add(MaxPooling2D((2, 2), strides=(2, 2)))

        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_1'))
        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_2'))
        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_3'))
        bottomModel.add(MaxPooling2D((2, 2), strides=(2, 2)))

        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_1'))
        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_2'))
        bottomModel.add(ZeroPadding2D((1, 1)))
        bottomModel.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_3'))
        #bottomModel.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="tf"))

        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call model.load_weights(filename)
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(bottomModel.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            bottomModel.layers[k].set_weights(weights)
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            layer = bottomModel.layers[k]
            if isinstance(layer, Convolution2D):
                weights[0] = np.transpose(np.array(weights[0])[:, :, ::-1, ::-1], (2, 3, 1, 0))
            layer.set_weights(weights)
        f.close()
        print('Model loaded.')


        self.model = Sequential()
        self.model.add(bottomModel)
        """"" self.model.add(Convolution2D(int(nb_filters / 2), kernel_size, padding='valid',
                                input_shape=input_shape, activation='relu',
                                kernel_initializer='he_normal', data_format = 'channels_first'))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Convolution2D(nb_filters, kernel_size, activation='relu',
                                kernel_initializer='he_normal', data_format = 'channels_first'))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        """
        self.model.add(Flatten())
        self.model.add(Dense(250, activation='relu', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(125, activation='relu', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(nb_classes, activation='softmax', kernel_initializer='he_normal'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
