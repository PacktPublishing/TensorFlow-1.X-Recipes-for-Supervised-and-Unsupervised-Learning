# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
Building and running the computational graph for:
    f(x,y) = x^2*y + 4*y
"""
import tensorflow as tf
from os.path import join

## 1. Create a directory to save TensorFlow logs ====
LOGDIR = './tf_logs/'

## 2. Give appropriate names to all graph nodes ====
# placeholders
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
mult_constant = tf.constant(4.0, name='c')
# operation nodes
f = x*x*y + mult_constant*y

# Create the session:
sess = tf.Session()

## Run the graph nodes
print("Program output:")
print(sess.run(f, feed_dict={x: 3, y: 2}))

## 3. Create a FileWriter object ====
writer = tf.summary.FileWriter(join(LOGDIR,'simple'), sess.graph)

## 4. Close the FileWriter object ====
writer.close()
sess.close()