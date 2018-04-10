from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

__C.anchors = [10, 13,  16, 30,  33, 23,  30, 61,  62, 45,  59, 119,  116, 90,  156, 198,  373, 326]
__C.classes = 20
__C.num = 9
#
# Training options
#
__C.train = edict()

__C.train.ignore_thresh = .5
__C.train.momentum = 0.9
__C.train.decay = 0.0005
__C.train.learning_rate = 0.001
__C.train.max_batches = 50200
__C.train.lr_steps = [40000, 45000]
__C.train.lr_scales = [.1, .1]
__C.train.mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

#
# image process options
#
__C.preprocess = edict()
__C.preprocess.angle = 0
__C.preprocess.saturation = 1.5
__C.preprocess.exposure = 1.5
__C.preprocess.hue = .1
__C.preprocess.jitter = .3
__C.preprocess.random = 1
