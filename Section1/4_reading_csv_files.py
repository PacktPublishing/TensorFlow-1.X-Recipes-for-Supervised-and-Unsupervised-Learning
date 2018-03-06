# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
"""

import tensorflow as tf

## 1. Create a tensor of type string with the file names
file_names = [".\\data\\file_" + str(x) + '.csv' for x in range(10)]
file_names = tf.constant(file_names, dtype=tf.string)

## 2. Create a dataset object, passing the tensor created
dataset = tf.contrib.data.TextLineDataset(file_names)

## 3. (Optional) Suffle the dataset
dataset = dataset.shuffle(buffer_size=1000)

## 4. (Optional) Create a batched dataset
dataset = dataset.batch(50) ## 20 batches to read all data

## 5. Create an Iterator object
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

## 6. Parse string tensors
## Provide default values
defaults = [[1], [1], [1], [1], [0]]
## Use decode_csv to get the values for each column of the file
col1, col2, col3, col4, target = tf.decode_csv(next_element,
                             record_defaults=defaults)
## stacking the columns into a single Tensor
features = tf.stack([col1, col2, col3, col4], axis=1)
target = tf.reshape(target, shape=[tf.shape(target)[0], 1])

## 7. Use the tensors in your computational graph
## creating a matrix multiplication node
mat_mul = tf.matmul(features, target, transpose_a=True)

with tf.Session() as sess:
    i = 0
    while True:
        try:
            print("=============BATCH {}============".format(i))
            print(sess.run(mat_mul))
            i+=1
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break