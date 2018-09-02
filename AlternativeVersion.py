import numpy as np
from mnist import MNIST
from keras.utils import np_utils
import cnn_nuovo

def main(train):
    # Carica i dati di emnist
    trainingData, trainingLabels, testData, testLabels, mapping = loadData('emnist')
    nn = cnn_nuovo.NeuralNetwork()

    if train:
        nn.train(trainingData, trainingLabels, epochs=10)
        nn.saveModel()   #viene eseguito il salvataggio del modello
    else:
        #nel caso in cui bisogna riprendere l'esecuzione da un certo punto carica il modello dall'apposito file
        try:
            nn.loadModel()
        except:
            print('[Error] No trained CNN model found.')

    nn.model.summary()

    preds = nn.readText(testData, mapping)
    print(preds)

    #Si valuta la precisione raggiunta dal modello
    nn.evaluate(trainingData, trainingLabels)

def loadData(path, ):
    # Vengono caricati dai rispettivi file i training set ed i test set
    emnistLoader = MNIST(path)
    trainingData, trainingLabels = emnistLoader.load(path + '/emnist-byclass-train-images-idx3-ubyte',
                                   path + '/emnist-byclass-train-labels-idx1-ubyte')
    testData, testLabels = emnistLoader.load(path + '/emnist-byclass-test-images-idx3-ubyte',
                                 path + '/emnist-byclass-test-labels-idx1-ubyte')

    # Si convertono i valori ACII in caratteri
    mapping = []

    with open(path + '/emnist-byclass-mapping.txt') as f:
        for line in f:
            mapping.append(chr(int(line.split()[1])))

    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    testData = np.array(testData)
    testLabels = np.array(testLabels)

    trainingData = normalize(trainingData)
    testData = normalize(testData)

    trainingData = reshape(trainingData)
    testData = reshape(testData)

    trainingLabels = preprocess_labels(trainingLabels, len(mapping))
    testLabels = preprocess_labels(testLabels, len(mapping))

    return trainingData, trainingLabels, testData, testLabels, mapping


def normalize(array):
    #Si trasorma un array in cui i dati sono degli interi compresi da 0 a 255 ad un array in cui i dati sono numeri decimali compresi tra 0 e 1
    array = array.astype('float32')
    array /= 255
    return array


def reshape(array, channels=1, width=28, height=28):
    #Viene eseguita la reshape delle immagini in modo che queste possano essere usate nella rete
    return array.reshape(array.shape[0], channels, width, height)


def preprocess_labels(array, nb_classes):
    # L'array di etichette viene trasformato in un vettore one-hot
    return np_utils.to_categorical(array, nb_classes)


if __name__ == '__main__':
    main(train=True)


