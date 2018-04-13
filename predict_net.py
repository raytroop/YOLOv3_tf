from yolo_top import yolov3
import numpy as np
import tensorflow as tf
from config import cfg
from PIL import Image, ImageDraw, ImageFont
from draw_boxes import draw_boxes
import matplotlib.pyplot as plt


# IMG_ID ='001542'
# image_test = Image.open('/home/raytroop/Dataset4ML/VOC2007/VOCdevkit/VOC2007/JPEGImages/{}.jpg'.format(IMG_ID))
image_test = Image.open('image/dog.jpg')
resized_image = image_test.resize((416, 416), Image.BICUBIC)
image_data = np.array(resized_image, dtype='float32')

imgs_holder = tf.placeholder(tf.float32, shape=[1, 416, 416, 3])
istraining = tf.constant(False, tf.bool)
cfg.batch_size = 1
cfg.scratch = True

model = yolov3(imgs_holder, None, istraining)
img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
boxes, scores, classes = model.pedict(img_hw, iou_threshold=0.5, score_threshold=0.5)

saver = tf.train.Saver()
ckpt_dir = './ckpt/'

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    boxes_, scores_, classes_ = sess.run([boxes, scores, classes],
                                         feed_dict={
                                                    img_hw: [image_test.size[1], image_test.size[0]],
                                                    imgs_holder: np.reshape(image_data / 255, [1, 416, 416, 3])})

    image_draw = draw_boxes(np.array(image_test, dtype=np.float32) / 255, boxes_, classes_, cfg.names, scores=scores_)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image_draw)
    fig.savefig('prediction.jpg')
    plt.show()

