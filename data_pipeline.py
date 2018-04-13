import tensorflow as tf
import matplotlib.pyplot as plt
from config import cfg
import numpy as np

def parser(example):
    features = {
                'xywhc': tf.FixedLenFeature([150], tf.float32),
                'img': tf.FixedLenFeature((), tf.string)}
    feats = tf.parse_single_example(example, features)
    coord = feats['xywhc']
    coord = tf.reshape(coord, [30, 5])

    img = tf.decode_raw(feats['img'], tf.float32)
    img = tf.reshape(img, [416, 416, 3])
    img = tf.image.resize_images(img, [cfg.train.image_resized, cfg.train.image_resized])
    rnd = tf.less(tf.random_uniform(shape=[], minval=0, maxval=2), 1)

    def flip_img_coord(_img, _coord):
        zeros = tf.constant([[0, 0, 0, 0, 0]]*30, tf.float32)
        img_flipped = tf.image.flip_left_right(_img)
        idx_invalid = tf.reduce_all(tf.equal(coord, 0), axis=-1)
        coord_temp = tf.concat([tf.minimum(tf.maximum(1 - _coord[:, :1], 0), 1),
                               _coord[:, 1:]], axis=-1)
        coord_flipped = tf.where(idx_invalid, zeros, coord_temp)
        return img_flipped, coord_flipped

    img, coord = tf.cond(rnd, lambda: (tf.identity(img), tf.identity(coord)), lambda: flip_img_coord(img, coord))

    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.minimum(img, 1.0)
    img = tf.maximum(img, 0.0)
    return img, coord


def data_pipeline(file_tfrecords, batch_size):
    dt = tf.data.TFRecordDataset(file_tfrecords)
    dt = dt.map(parser, num_parallel_calls=4)
    dt = dt.prefetch(batch_size)
    dt = dt.shuffle(buffer_size=20*batch_size)
    dt = dt.repeat()
    dt = dt.batch(batch_size)
    iterator = dt.make_one_shot_iterator()
    imgs, true_boxes = iterator.get_next()

    return imgs, true_boxes


if __name__ == '__main__':
    file_path = 'trainval0712.tfrecords'
    imgs, true_boxes = data_pipeline(file_path, cfg.batch_size)
    sess = tf.Session()
    imgs_, true_boxes_ = sess.run([imgs, true_boxes])
    print(imgs_.shape, true_boxes_.shape)
    for imgs_i, boxes_ in zip(imgs_, true_boxes_):
        valid = (np.sum(boxes_, axis=-1) > 0).tolist()
        print([cfg.names[int(idx)] for idx in boxes_[:, 4][valid].tolist()])
        plt.figure()
        plt.imshow(imgs_i)
    plt.show()