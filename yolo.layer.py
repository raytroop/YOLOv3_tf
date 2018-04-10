import tensorflow as tf
import numpy as np

feat_74 = tf.placeholder(dtype=tf.float32, shape=(3, 13, 13, 75))

txy0 = feat_74[..., 0:2]
txy1 = feat_74[..., 25:27]
txy2 = feat_74[..., 50:52]

twh0 = feat_74[..., 2:4]
twh1 = feat_74[..., 27:29]
twh2 = feat_74[..., 52:54]

to0 = feat_74[..., 4:5]
to1 = feat_74[..., 29:30]
to2 = feat_74[..., 54:55]

cls0 = feat_74[..., 5:25]
cls1 = feat_74[..., 30:50]
cls2 = feat_74[..., 55:75]


logit_txy0 = tf.nn.sigmoid(txy0)
logit_txy1 = tf.nn.sigmoid(txy1)
logit_txy2 = tf.nn.sigmoid(txy2)
logit_txy = tf.concat([logit_txy0, logit_txy1, logit_txy2], axis=-1)
logit_txy = tf.reshape(logit_txy, [-1, 2])

twh = tf.concat([twh0, twh1, twh2], axis=-1)
twh = tf.reshape(twh, [-1, 2])

logit_to0 = tf.nn.sigmoid(to0)
logit_to1 = tf.nn.sigmoid(to1)
logit_to2 = tf.nn.sigmoid(to2)
logit_to = tf.concat([logit_to0, logit_to1, logit_to2], axis=-1)
logit_to = tf.reshape(logit_to, [-1, 1])

logit_cls0 = tf.nn.sigmoid(cls0)
logit_cls1 = tf.nn.sigmoid(cls1)
logit_cls2 = tf.nn.sigmoid(cls2)
logit_cls = tf.concat([logit_cls0, logit_cls1, logit_cls2], axis=-1)
logit_cls = tf.reshape(logit_cls, [-1, 20])

lhw = tf.shape(feat_74)[1:3]
cx, cy = tf.meshgrid(tf.range(lhw[1]), tf.range(lhw[0]))
cxy = tf.stack([cx, cy], axis=2)
cxy = tf.reshape(cxy, shape=[-1, 2])
bx=
sess = tf.Session()
# print(sess.run([x_grid, y_grid], feed_dict={feat_74: np.ones([25, 13, 13, 75])}))

# // box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
# //{
# //    box b;
# //    b.x = (i + x[index + 0*stride]) / lw;     // relative to [the whole feature map shape], (σ(tx) + cx) / lw
# //    b.y = (j + x[index + 1*stride]) / lh;     // relative to [the whole feature map shape], (σ(ty) + cy) / lh
# //    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;   // relative to [input image shape], pw*exp(tw) / w
# //    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;   // relative to [input image shape], ph*exp(th) / h
# //    return b;
# //}
