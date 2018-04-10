import configparser
import io
import numpy as np
from collections import defaultdict
from collections import namedtuple

weights_path = "darknet53.conv.74"
config_path = "yolov3-voc.cfg"
conv_bn = namedtuple('conv_bn', ['bias', 'gamma', 'mean', 'variance', 'conv_weights'])
weights_dict = {}


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


unique_config_file = unique_config_sections(config_path)
cfg_parser = configparser.ConfigParser()
cfg_parser.read_file(unique_config_file)
weights_file = open(weights_path, 'rb')
major = np.ndarray(shape=(1, ), dtype='int32', buffer=weights_file.read(4))
minor = np.ndarray(shape=(1, ), dtype='int32', buffer=weights_file.read(4))
revision = np.ndarray(shape=(1, ), dtype='int32', buffer=weights_file.read(4))
seen = np.ndarray(shape=(1, ), dtype=np.int64, buffer=weights_file.read(8))

filter_list = [3, ]

for section in cfg_parser.sections():
    if len(filter_list) > 74:
        break
    print('Parsing section {}'.format(section))
    if section.startswith('convolutional'):
        filters = int(cfg_parser[section]['filters'])
        size = int(cfg_parser[section]['size'])
        stride = int(cfg_parser[section]['stride'])
        pad = int(cfg_parser[section]['pad'])
        activation = cfg_parser[section]['activation']
        batch_normalize = 'batch_normalize' in cfg_parser[section]
        # padding='same' is equivalent to Darknet pad=1
        padding = 'same' if pad == 1 else 'valid'
        # Setting weights.
        # Darknet serializes convolutional weights as:
        # [bias/beta, [gamma, mean, variance], conv_weights]

        # TODO: This assumes channel last dim_ordering.
        weights_shape = (size, size, filter_list[-1], filters)
        darknet_w_shape = (filters, weights_shape[2], size, size)
        weights_size = np.product(weights_shape)
        print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)

        conv_bias = np.ndarray(
            shape=(filters,),
            dtype='float32',
            buffer=weights_file.read(filters * 4))

        if batch_normalize:
            bn_weights = np.ndarray(  # gamma0, gamma1, ... mean0, mean0, ... , var0, var0, ...
                shape=(3, filters),
                dtype='float32',
                buffer=weights_file.read(filters * 12))

            # TODO: Keras BatchNormalization mistakenly refers to var
            # as std.
            bn_weight_list = [
                bn_weights[0],  # scale gamma
                conv_bias,  # shift beta
                bn_weights[1],  # running mean
                bn_weights[2]  # running var
            ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            # TODO: Add check for Theano dim ordering.
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias]

        filter_list.append(filters)
        weights = conv_bn(bias=conv_bias, gamma=bn_weights[0], mean=bn_weights[1], variance=bn_weights[2],
                          conv_weights=conv_weights)
        section = 'conv_' + section.split('_')[-1]
        weights_dict[section] = weights

    elif section.startswith('shortcut'):
        from_list = [int(l) for l in (cfg_parser[section]['from']).strip().split(',')]
        assert from_list[0] < 0, 'relative coord'
        c_ = filter_list[from_list[0]]
        print('shortcut #channel:{}'.format(c_))
        filter_list.append(c_)

    elif section.startswith('net'):
        pass
    else:
        raise ValueError(
            'Unsupported section header type: {}'.format(section))
weights_file.close()
np.savez('darknet53.conv.74.npz', **weights_dict)