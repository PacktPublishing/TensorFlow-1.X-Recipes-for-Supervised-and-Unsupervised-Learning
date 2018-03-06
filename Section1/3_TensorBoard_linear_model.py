# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
"""
import tensorflow as tf
from os.path import join

## 1. Create a directory to save TensorFlow logs ====
LOGDIR = './tf_logs/'

## 2. Give appropriate names to all graph nodes ====
# Model parameters
W = tf.Variable(0.0, dtype=tf.float32, name='W')
b = tf.Variable(0.0, dtype=tf.float32, name='intercept')

# Model input and output
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

# Trainable model
linear_model = W * x + b
# Creating a namespace for training operations
with tf.name_scope("training"):
    #loss
    loss = tf.reduce_sum(tf.square(linear_model - y), name='loss') # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01, name='opt')
    train = optimizer.minimize(loss, name='train')

# training data
x_train = [0., 1., 2., 3., 4., 5.] 
y_train = [-1.0, 0.5, 1.1, 2.6, 3.5, 4.0]

# initializer node
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
# training loop
for i in range(10):
  sess.run(train, feed_dict={x: x_train, y: y_train})
  print("W: ", sess.run(W), "b: ", sess.run(b))
  
## 3. Create a FileWriter object ====
writer = tf.summary.FileWriter(join(LOGDIR,'linear_model'), sess.graph)

## 4. Close the FileWriter object ====
writer.close()

print("Final W value: ", sess.run(W))
print("Final intercept value: ", sess.run(b))

sess.close()