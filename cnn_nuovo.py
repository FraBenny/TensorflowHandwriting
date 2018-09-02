import numpy as np
from os import getcwd
from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
import h5py


class NeuralNetwork(object):

    def __init__(self):
        self.model = None

    def train(self, data, labels, batch_size=128, epochs=10):
        #Il modello viene allenato
        self.getModel()
        callbacks.Callback()
        checkpoint = callbacks.ModelCheckpoint(filepath=getcwd()+'weights.{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        #self.model.load_weights(filepath=getcwd()+'weights.04.hdf5', by_name=False)
        self.model.fit(data, labels, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
        print(checkpoint)

    def evaluate(self, data, labels):
        #Viene valutata l'accuratezza del modello che è stato allenato
        score = self.model.evaluate(data, labels)
        print('Loss:', score[0])
        print('Accuracy:', score[1])

    def saveModel(self):
        #Viene salvato il modello allenato
        self.model.save('emnist-cnn.h5', overwrite=True)

    def loadModel(self):
        #Viene caricato il modello allenato
        self.model = load_model('emnist-cnn.h5')

    def readText(self, data, mapping):
        #Vengono individuati i caratteri scritti a mano nell'immagine
        prediction = self.model.predict(data)
        prediction = np.argmax(prediction, axis=1)
        return ''.join(mapping[x] for x in prediction)

    def getModel(self, classes=62, filters=32, kernel_size=(5, 5), pool_size=(2, 2), input_shape=(1, 28, 28)):
        """Viene costruito il modello della rete neurale convoluzionale. Vengono passati come parametri il numero di classi e di filtri
        da utilizzare, le dimensione del filtro, le dimensioni per il max pooling e la forma dell'input (channel_first)
        """
        self.model = Sequential()
        self.model.add(Convolution2D(int(filters / 2), kernel_size, padding='valid',
                                input_shape=input_shape, activation='relu',
                                kernel_initializer='he_normal', data_format = 'channels_first'))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Convolution2D(filters, kernel_size, activation='relu',
                                kernel_initializer='he_normal', data_format = 'channels_first'))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(classes, activation='softmax', kernel_initializer='he_normal'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
