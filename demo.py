import tensorflow as tf
import numpy as np
import itertools
import cv2
import time
from scipy import ndimage
from TianChiDateReader import TianChiDateReader
from model import Layers


Feed_dict = {}
Ret_dict = {}
Layers()
STAGE = 2
rt = TianChiDateReader("../train_modified", train_size=0.95)

images, landmarks, visibilities = rt.get_validation_data()
with tf.Session() as Sess:
    Saver = tf.train.Saver()
    Writer = tf.summary.FileWriter("logs/", Sess.graph)
    if STAGE == 0:
        Sess.run(tf.global_variables_initializer())
    else:
        Saver.restore(Sess, './Model/Model')
        print('Model Read Over!')

    S1_Landmark,d_landmark=Sess.run([Ret_dict['S1_Landmark'], Ret_dict['d_landmark']],
             {Feed_dict['InputImage']: images, Feed_dict['Visible']: visibilities,
              Feed_dict['S1_isTrain']: False, Feed_dict['S2_isTrain']: False})