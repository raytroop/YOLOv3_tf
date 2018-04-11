import tensorflow as tf
from darknet53_trainable import Darknet53
import cv2
import numpy as np
model = Darknet53(darknet53_npz_path='darknet53.conv.74.npz')
img_holder = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='img')
phase_train = tf.placeholder(tf.bool, shape=(), name='phase_train')
feat_ex = model.build(img_holder, phase_train)
bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv_0/bias")[0]
gamma = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv_0/gamma")[0]
variance = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv_0/variance")[0]
mean = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv_0/mean")[0]
print(bias)
# for var in tf.global_variables():
#     print(var)
#
# print(len(tf.trainable_variables()))
# print(model.conv3)
img = cv2.imread('bladerrunner.jpg')
img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
img = np.expand_dims(img, axis=0)
# print(img.shape)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
out_ = sess.run(feat_ex, feed_dict={phase_train: True, img_holder: img})
print(out_.shape)
bias_, gamma_, variance_, mean_ = sess.run([bias, gamma, variance, mean])
print(bias_)
print(gamma_)
print(mean_)
print(variance_)