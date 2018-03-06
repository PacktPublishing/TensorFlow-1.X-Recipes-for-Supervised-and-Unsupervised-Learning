# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
Building and running the computational graph for:
    f(x,y) = x^2*y + 4*y
"""

import tensorflow as tf

## 1. Define all nodes in the graph ====
# placeholders
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# operation nodes
square_node = x*x
mult_node = square_node*y
quadruple_node = 4*y
adder_node = mult_node + quadruple_node

#
#f =x*x*y + 4*y
## 2. Create a TensorFlow session object ====
sess = tf.Session()

## 3. Initialize all variables (if any) ====

## 4. Run the nodes that produce the results ====
print("\nProgram output:")
print(sess.run(adder_node, feed_dict={x: 3, y: 2}))

# 5. Close the session ====
sess.close()