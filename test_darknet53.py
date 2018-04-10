import tensorflow as tf
import numpy as np
from darknet53_trainable import Darknet53

model = Darknet53(darknet53_npz_path='darknet53.conv.74.npz')
img = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='img')
phase_train = tf.placeholder(tf.bool, shape=(), name='phase_train')
feat_ex = model.build(img, phase_train)
for var in tf.global_variables():
    print(var)

print(len(tf.trainable_variables()))