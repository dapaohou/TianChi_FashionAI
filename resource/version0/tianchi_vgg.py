import tensorflow as tf
import numpy as np

from tensorflow_vgg.vgg16 import Vgg16
from TianChiDateReader import TianChiDateReader

num_landmarks = 24

input_ = tf.placeholder(tf.float32, [None, 224, 224, 3], name="input_")
ground_truth_int = tf.placeholder(tf.int64, [None, num_landmarks*2], name="ground_truth_int")
visible_int = tf.placeholder(tf.int64, [None, num_landmarks*2], name="visibilities_int")
isTrain = tf.placeholder(tf.bool, name="isTrain")

ground_truth = tf.cast(ground_truth_int, dtype=tf.float32)      # covert to float
visible = tf.cast(visible_int, dtype=tf.float32)                # covert to float

fc6_result = tf.placeholder(tf.float32, [None, 4096], name="fc6_result")    # result from vgg16

with tf.name_scope("fc"):
    fc8 = tf.layers.batch_normalization(
        tf.layers.dense(fc6_result, 512, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer()),
        training=isTrain)
    fc9 = tf.layers.batch_normalization(
        tf.layers.dense(fc8, num_landmarks*2, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer()),
        training=isTrain)
with tf.name_scope("loss"):
    mask = (visible + tf.multiply(visible, visible)) / 2
    loss_matrix = tf.multiply(tf.square(fc9 - ground_truth), mask)
    loss = tf.reduce_mean(tf.reduce_sum(loss_matrix, axis=1, keep_dims=True))
with tf.name_scope("optimizer"):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'fc')):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tr = TianChiDateReader("../train_modified", img_size=224, train_size=0.8)
    vgg = Vgg16()
    with tf.name_scope("Vgg"):
        vgg.build(input_)
    for e in range(3):
        for images, landmarks, visibilities in tr.get_train_data(batch_size=10):
            fc6_ = sess.run(vgg.relu6, feed_dict={input_: images})
            sess.run(optimizer, feed_dict={fc6_result: fc6_, ground_truth_int: landmarks, visible_int: visibilities, isTrain: True})
            print(sess.run(loss, feed_dict={fc6_result: fc6_, ground_truth_int: landmarks, visible_int: visibilities, isTrain: True}))


