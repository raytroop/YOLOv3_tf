from yolo_head import yolo_head
import tensorflow as tf


feat_52 = tf.placeholder(dtype=tf.float32, shape=(None, 13, 13, 1024))
res18 = tf.placeholder(dtype=tf.float32, shape=(None, 26, 26, 512))
res10 = tf.placeholder(dtype=tf.float32, shape=(None, 52, 52, 256))
istraining = tf.placeholder(dtype=tf.bool, shape=[])
head = yolo_head(istraining)
feat0, feat1, feat2 = head.build(feat_52, res18, res10)
print(feat0)
print(feat1)
print(feat2)

# Tensor("conv_head_74/conv2d/BiasAdd:0", shape=(?, 52, 52, 75), dtype=float32)
# Tensor("conv_head_66/conv2d/BiasAdd:0", shape=(?, 26, 26, 75), dtype=float32)
# Tensor("conv_head_58/conv2d/BiasAdd:0", shape=(?, 13, 13, 75), dtype=float32)