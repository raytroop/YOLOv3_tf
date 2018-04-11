import tensorflow as tf
from yolo_layer import preprocess_true_boxes
import numpy as np

image_size = tf.constant([200, 416, 416, 3], tf.int32)
feat_size = tf.constant([8, 13, 13, 75], tf.int32)
anchors = tf.constant([[10, 13], [16, 30], [33, 23]], tf.float32)
true_boxes = tf.constant(np.abs(np.random.normal(0, 1, [8, 30, 5])), tf.float32)

loc_cls, mask, scale = preprocess_true_boxes(true_boxes, anchors, feat_size, image_size)
print(loc_cls)
print(mask)
sess = tf.Session()
loc_cls_, mask_, scale_ = sess.run([loc_cls, mask, scale])
print(loc_cls_.shape)
print(mask_.shape)
print(scale_.shape)