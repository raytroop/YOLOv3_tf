import tensorflow as tf
import numpy as np

feat1 = np.arange(9).reshape(3, 3)
feat2 = np.arange(9, 18).reshape(3, 3)
feat_map = np.stack([feat1, feat2], axis=2)
feat_map = np.expand_dims(feat_map, 0)
# print(feat_map[:, :, 1])
feats = tf.constant(feat_map, dtype=tf.float32)
new_feats = tf.image.resize_nearest_neighbor(feats, [6, 6])
sess = tf.Session()
new_feats_ = sess.run(new_feats)
print(new_feats_[0, :, :, 0].shape)
print(new_feats_[0, :, :, 1])