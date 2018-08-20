from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  X = tf.placeholder("float", [None, n_input])
  Y = tf.placeholder("float", [None, n_classes])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  weights = {
      'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
      'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
      'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
  }
  biases = {
      'b1': tf.Variable(tf.random_normal([n_hidden_1])),
      'b2': tf.Variable(tf.random_normal([n_hidden_2])),
      'out': tf.Variable(tf.random_normal([n_classes]))
  }

  # Hidden fully connected layer with 256 neurons
  layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
  # Hidden fully connected layer with 256 neurons
  layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
  # Output fully connected layer with a neuron for each class
  out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
  # The raw formulation of cross-entropy,
  #

  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=out_layer)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(
      accuracy, feed_dict={
          X: mnist.test.images,
          y_: mnist.test.labels
      }))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
