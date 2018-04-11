import tensorflow as tf
from darknet53_trainable import Darknet53
from yolo_layer import yolo_head, yolo_det, preprocess_true_boxes, confidence_loss, cord_cls_loss
from config import cfg


class yolov3:

    def __init__(self, img, truth, istraining, decay_bn=0.99):
        self.img = img
        self.truth = truth
        self.istraining = istraining
        self.decay_bn = decay_bn
        self.img_shape = tf.shape(self.img)

    # def build(self):
        with tf.variable_scope("Feature_Extractor"):
            feature_extractor = Darknet53('darknet53.conv.74.npz', scratch=cfg.scratch)
            self.feats52 = feature_extractor.build(self.img, self.istraining, self.decay_bn)
            head = yolo_head(self.istraining)
            self.yolo123, self.yolo456, self.yolo789 = head.build(self.feats52,
                                                                  feature_extractor.res18,
                                                                  feature_extractor.res10)
        with tf.variable_scope("Detection_0"):
            self.anchors0 = tf.constant(cfg.anchors[cfg.train.mask[0]], dtype=tf.float32)
            det = yolo_det(self.anchors0, cfg.classes, self.img_shape)
            self.pred_xy0, self.pred_wh0, self.pred_confidence0, self.pred_class_prob0, self.loc_txywh0 = det.build(self.yolo123)
        with tf.variable_scope("Detection_1"):
            self.anchors1 = tf.constant(cfg.anchors[cfg.train.mask[1]], dtype=tf.float32)
            det = yolo_det(self.anchors1, cfg.classes, self.img_shape)
            self.pred_xy1, self.pred_wh1, self.pred_confidence1, self.pred_class_prob1, self.loc_txywh1 = det.build(self.yolo456)
        with tf.variable_scope("Detection_2"):
            self.anchors2 = tf.constant(cfg.anchors[cfg.train.mask[2]], dtype=tf.float32)
            det = yolo_det(self.anchors2, cfg.classes, self.img_shape)
            self.pred_xy2, self.pred_wh2, self.pred_confidence2, self.pred_class_prob2, self.loc_txywh2 = det.build(self.yolo789)


    def compute_loss(self):
        with tf.name_scope('Loss_0'):
            matching_true_boxes, detectors_mask, loc_scale = preprocess_true_boxes(self.truth,
                                                                                   self.anchors0,
                                                                                   tf.shape(self.yolo123),
                                                                                   self.img_shape)
            objectness_loss = confidence_loss(self.pred_xy0, self.pred_wh0, self.pred_confidence0, self.truth, detectors_mask)
            cord_loss = cord_cls_loss(detectors_mask, matching_true_boxes,
                                      cfg.classes, self.pred_class_prob0, self.loc_txywh0, loc_scale)
            loss1 = objectness_loss + cord_loss
        with tf.name_scope('Loss_1'):
            matching_true_boxes, detectors_mask, loc_scale = preprocess_true_boxes(self.truth,
                                                                                   self.anchors1,
                                                                                   tf.shape(self.yolo456),
                                                                                   self.img_shape)
            objectness_loss = confidence_loss(self.pred_xy1, self.pred_wh1, self.pred_confidence1, self.truth, detectors_mask)
            cord_loss = cord_cls_loss(detectors_mask, matching_true_boxes,
                                      cfg.classes, self.pred_class_prob1, self.loc_txywh1, loc_scale)
            loss2 = objectness_loss + cord_loss
        with tf.name_scope('Loss_2'):
            matching_true_boxes, detectors_mask, loc_scale = preprocess_true_boxes(self.truth,
                                                                                   self.anchors2,
                                                                                   tf.shape(self.yolo789),
                                                                                   self.img_shape)
            objectness_loss = confidence_loss(self.pred_xy2, self.pred_wh2, self.pred_confidence2, self.truth, detectors_mask)
            cord_loss = cord_cls_loss(detectors_mask, matching_true_boxes,
                                      cfg.classes, self.pred_class_prob2, self.loc_txywh2, loc_scale)
            loss3 = objectness_loss + cord_loss

        self.loss = loss1 + loss2 + loss3
        return self.loss

    def pedict(self, iou_threshold=0.5):
        """
        For now, only support single image prediction
        :param iou_threshold:
        :return:
        """
        with tf.name_scope('Predict_0'):
            pred_loc0 = tf.concat([self.pred_xy0[..., 1:] - 0.5 * self.pred_wh0[..., 1:],
                                   self.pred_xy0[..., 0:1] - 0.5 * self.pred_wh0[..., 0:1],
                                   self.pred_xy0[..., 1:] + 0.5 * self.pred_wh0[..., 1:],
                                   self.pred_xy0[..., 0:1] + 0.5 * self.pred_wh0[..., 0:1]
                                   ], axis=-1)  # (y1, x1, y2, x2)
            pred_loc0 = tf.maximum(tf.minimum(pred_loc0, 1), 0)
            pred_loc0 = tf.reshape(pred_loc0, [-1, 4]) * \
                        tf.expand_dims(tf.stack([self.img_shape[1], self.img_shape[2]] * 2, axis=0), axis=0)
            pred_obj0 = tf.reshape(self.pred_confidence0, shape=[-1])
            pred_cls0 = tf.reshape(self.pred_class_prob0, [-1, cfg.classes])
        with tf.name_scope('Predict_1'):
            pred_loc1 = tf.concat([self.pred_xy1[..., 1:] - 0.5 * self.pred_wh1[..., 1:],
                                   self.pred_xy1[..., 0:1] - 0.5 * self.pred_wh1[..., 0:1],
                                   self.pred_xy1[..., 1:] + 0.5 * self.pred_wh1[..., 1:],
                                   self.pred_xy1[..., 0:1] + 0.5 * self.pred_wh1[..., 0:1]
                                   ], axis=-1)  # (y1, x1, y2, x2)
            pred_loc1 = tf.maximum(tf.minimum(pred_loc1, 1), 0)
            pred_loc1 = tf.reshape(pred_loc1, [-1, 4]) * \
                        tf.expand_dims(tf.stack([self.img_shape[1], self.img_shape[2]] * 2, axis=0), axis=0)
            pred_obj1 = tf.reshape(self.pred_confidence1, shape=[-1])
            pred_cls1 = tf.reshape(self.pred_class_prob1, [-1, cfg.classes])
        with tf.name_scope('Predict_2'):
            pred_loc2 = tf.concat([self.pred_xy2[..., 1:] - 0.5 * self.pred_wh2[..., 1:],
                                   self.pred_xy2[..., 0:1] - 0.5 * self.pred_wh2[..., 0:1],
                                   self.pred_xy2[..., 1:] + 0.5 * self.pred_wh2[..., 1:],
                                   self.pred_xy2[..., 0:1] + 0.5 * self.pred_wh2[..., 0:1]
                                   ], axis=-1)  # (y1, x1, y2, x2)
            pred_loc2 = tf.maximum(tf.minimum(pred_loc2, 1), 0)
            pred_loc2 = tf.reshape(pred_loc2, [-1, 4]) * \
                        tf.expand_dims(tf.stack([self.img_shape[1], self.img_shape[2]] * 2, axis=0), axis=0)
            pred_obj2 = tf.reshape(self.pred_confidence2, shape=[-1])
            pred_cls2 = tf.reshape(self.pred_class_prob2, [-1, cfg.classes])

        self.pred_loc = tf.concat([pred_loc0, pred_loc1, pred_loc2], axis=0, name='pred_y1x1y2x2')
        self.pred_obj = tf.concat([pred_obj0, pred_obj1, pred_obj2], axis=0, name='pred_objectness')
        self.pred_cls = tf.concat([pred_cls0, pred_cls1, pred_cls2], axis=0, name='pred_clsprob')

        sel_idx = tf.image.non_max_suppression(self.pred_loc, self.pred_obj,
                                               max_output_size=5,
                                               iou_threshold=iou_threshold)
        sel_loc = tf.gather(self.pred_loc, sel_idx)
        sel_obj = tf.gather(self.pred_obj, sel_idx)
        sel_cls_prob = tf.reduce_max(tf.gather(self.pred_cls, sel_idx), axis=-1)
        sel_cls_label = tf.argmax(tf.gather(self.pred_cls, sel_idx), axis=-1)

        return sel_obj, sel_loc, sel_cls_prob, sel_cls_label






