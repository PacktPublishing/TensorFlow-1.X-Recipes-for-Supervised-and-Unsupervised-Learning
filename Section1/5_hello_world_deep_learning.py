# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
"""

import tensorflow as tf
from numpy import argmax

## 1. Read the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/")

## 2. Build the fully connected layer function
def fully_connected_layer(input_tensor, n_neurons, activation=True):
    '''Produces a fully connected network layer with random normal
    initialization and relu activation function'''
    n_inputs = int(input_tensor.get_shape()[1])
    W = tf.Variable(tf.random_normal(shape=(n_inputs, n_neurons), stddev=0.1))
    b = tf.Variable(tf.zeros([n_neurons]))
    Z = tf.matmul(input_tensor, W) + b
    if activation:
        return tf.nn.relu(Z)
    else:
        return Z


## 3. Decide the architecture of your fully connected DNN 
n_inputs = 28*28  # MNIST image dimensions
n_hidden1 = 350
n_hidden2 = 200 
n_hidden3 = 100
n_outputs = 10 # [0,0,0,0,0,1,0,0,0]

# 4. Define placeholders for inputs and labels
# input layer 
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# target values
y = tf.placeholder(tf.int64)

# 5. Build the DNN
hidden1 = fully_connected_layer(X, n_hidden1)
hidden2 = fully_connected_layer(hidden1, n_hidden2)
hidden3 = fully_connected_layer(hidden2, n_hidden3)
logits = fully_connected_layer(hidden3, n_outputs, activation=False)

## 6. Define the loss function
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                   labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)


## 7. Set the optimization parameters: epochs, mini-batch size, learning rate
n_epochs = 20
batch_size = 80
learning_rate = 0.01

## 8. Define the optimizer and the training operation
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
    
## 9. Evaluate the accuracy of the networks 
correct = tf.nn.in_top_k(predictions=logits, targets=y, k=1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
## 10. Run the computational graph
 
with tf.Session() as sess:
    ## Initializing the variables
    tf.global_variables_initializer().run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print("Epoch:", epoch+1)
        print("Train accuracy:", acc_train, "| Test accuracy:", acc_test)
        print(50*"-")
    print("Done Trainning!")

    ## Producing individual predictions
    print("\n=====================\n")
    print("Using the network to make individual predictions")
    n_pred = 15
    X_new = mnist.test.images[:n_pred]
    Z = logits.eval(feed_dict={X: X_new})
    y_pred = argmax(Z, axis=1)
    print("Actual | Predicted")
    print("=====================")
    for obs, pred in zip(mnist.test.labels[:n_pred], y_pred):
        print("{: >4}   |{: >6}".format(obs, pred))


