from yolo_top import yolov3
import numpy as np
import tensorflow as tf
from data_pipeline import data_pipeline
from config import cfg

file_path = 'trainval0712.tfrecords'
imgs, true_boxes = data_pipeline(file_path, cfg.batch_size)

istraining = tf.constant(True, tf.bool)
model = yolov3(imgs, true_boxes, istraining)

loss = model.compute_loss()
global_step = tf.Variable(0, trainable=False)
# lr = tf.train.exponential_decay(0.0001, global_step=global_step, decay_steps=2e4, decay_rate=0.1)
lr = tf.train.piecewise_constant(global_step, [40000, 45000], [1e-3, 1e-4, 1e-5])
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
# optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Head")
# for var in vars_det:
#     print(var)
with tf.control_dependencies(update_op):
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_det)
saver = tf.train.Saver()
ckpt_dir = './ckpt/'

gs = 0
batch_per_epoch = 2000
cfg.train.max_batches = int(batch_per_epoch * 10)
cfg.train.image_resized = 608
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if(i % 1000 == 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)

cfg.train.max_batches = int(batch_per_epoch * 20)
cfg.train.image_resized = 512
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if(i % 1000 == 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)

cfg.train.max_batches = int(batch_per_epoch * 30)
cfg.train.image_resized = 320
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if(i % 1000 == 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)

cfg.train.max_batches = int(batch_per_epoch * 40)
cfg.train.image_resized = 352
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if(i % 1000 == 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)

cfg.train.max_batches = int(batch_per_epoch * 50)
cfg.train.image_resized = 480
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if(i % 1000 == 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)