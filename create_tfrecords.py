import xml.etree.ElementTree as ET
import numpy as np
import os
import tensorflow as tf
from PIL import Image


sets = [('2007', 'trainval'), ('2012', 'trainval')]


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]


def convert_annotation(year, image_id):
    in_file = open('/home/raytroop/Dataset4ML/VOC%s/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, year, image_id))

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    for i, obj in enumerate(root.iter('object')):
        if i > 29:
            break
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b) + [cls_id]
        bboxes.extend(bb)
    if len(bboxes) < 30*5:
        bboxes = bboxes + [0, 0, 0, 0, 0]*(30-int(len(bboxes)/5))

    return np.array(bboxes, dtype=np.float32).flatten().tolist()

def convert_img(year, image_id):
    image = Image.open('/home/raytroop/Dataset4ML/VOC%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (year, year, image_id))
    resized_image = image.resize((416, 416), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')/255
    img_raw = image_data.tobytes()
    return img_raw

filename = os.path.join('trainval'+'0712'+'.tfrecords')
writer = tf.python_io.TFRecordWriter(filename)
for year, image_set in sets:
    image_ids = open('/home/raytroop/Dataset4ML/VOC%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (
        year, year, image_set)).read().strip().split()
    # print(filename)
    for image_id in image_ids:
        xywhc = convert_annotation(year, image_id)
        img_raw = convert_img(year, image_id)

        example = tf.train.Example(features=tf.train.Features(feature={
            'xywhc':
                    tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
            'img':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))
        writer.write(example.SerializeToString())
writer.close()
