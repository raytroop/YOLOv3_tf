from yolo_top import yolov3
import numpy as np
import tensorflow as tf


img = tf.constant(np.random.normal(0, 1, [8, 416, 416, 3]), tf.float32)
truth = tf.constant(np.random.randint(0, 2, size=[8, 30, 5]), tf.float32)
istraining = tf.constant(True, tf.bool)
model = yolov3(img, truth, istraining)

loss = model.compute_loss()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(loss))