import tensorflow as tf


class yolo_head:

    def __init__(self, istraining):
        self.istraining = istraining

    def conv_layer(self, bottom, size, stride, in_channels, out_channels, use_bn, name):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(bottom, out_channels, size, stride, padding="SAME",
                                    use_bias=not use_bn, activation=None)
            if use_bn:
                conv_bn = tf.layers.batch_normalization(conv, training=self.istraining)
                act = tf.nn.leaky_relu(conv_bn, 0.1)
            else:
                act = conv
        return act
    def build(self, feat_ex, res18, res10):
        self.conv52 = self.conv_layer(feat_ex, 1, 1, 1024, 512, True, 'conv_head_52')  # 13x512
        self.conv53 = self.conv_layer(self.conv52, 3, 1, 512, 1024, True, 'conv_head_53')   # 13x1024
        self.conv54 = self.conv_layer(self.conv53, 1, 1, 1024, 512, True, 'conv_head_54')   # 13x512
        self.conv55 = self.conv_layer(self.conv54, 3, 1, 512, 1024, True, 'conv_head_55')   # 13x1024
        self.conv56 = self.conv_layer(self.conv55, 1, 1, 1024, 512, True, 'conv_head_56')   # 13x512
        self.conv57 = self.conv_layer(self.conv56, 3, 1, 512, 1024, True, 'conv_head_57')   # 13x1024
        self.conv58 = self.conv_layer(self.conv57, 1, 1, 1024, 75, False, 'conv_head_58')   # 13x75
        # follow yolo layer mask = 6,7,8
        self.conv59 = self.conv_layer(self.conv56, 1, 1, 512, 256, True, 'conv_head_59')    # 13x256
        size = tf.shape(self.conv59)[1]
        self.upsample0 = tf.image.resize_nearest_neighbor(self.conv59, [2*size, 2*size],
                                                          name='upsample_0')                # 26x256
        self.route0 = tf.concat([self.upsample0, res18], axis=-1, name='route_0')           # 26x768
        self.conv60 = self.conv_layer(self.route0, 1, 1, 768, 256, True, 'conv_head_60')    # 26x256
        self.conv61 = self.conv_layer(self.conv60, 3, 1, 256, 512, True, 'conv_head_61')    # 26x512
        self.conv62 = self.conv_layer(self.conv61, 1, 1, 512, 256, True, 'conv_head_62')    # 26x256
        self.conv63 = self.conv_layer(self.conv62, 3, 1, 256, 512, True, 'conv_head_63')    # 26x512
        self.conv64 = self.conv_layer(self.conv63, 1, 1, 512, 256, True, 'conv_head_64')    # 26x256
        self.conv65 = self.conv_layer(self.conv64, 3, 1, 256, 512, True, 'conv_head_65')    # 26x512
        self.conv66 = self.conv_layer(self.conv65, 1, 1, 512, 75, False, 'conv_head_66')    # 26x75
        # follow yolo layer mask = 3,4,5
        self.conv67 = self.conv_layer(self.conv64, 1, 1, 256, 128, True, 'conv_head_67')    # 26x128
        size = tf.shape(self.conv67)[1]
        self.upsample1 = tf.image.resize_nearest_neighbor(self.conv67, [2 * size, 2 * size],
                                                          name='upsample_1')                # 52x128
        self.route1 = tf.concat([self.upsample1, res10], axis=-1, name='route_1')           # 52x384
        self.conv68 = self.conv_layer(self.route1, 1, 1, 384, 128, True, 'conv_head_68')    # 52x128
        self.conv69 = self.conv_layer(self.conv68, 3, 1, 128, 256, True, 'conv_head_69')    # 52x256
        self.conv70 = self.conv_layer(self.conv69, 1, 1, 256, 128, True, 'conv_head_70')    # 52x128
        self.conv71 = self.conv_layer(self.conv70, 3, 1, 128, 256, True, 'conv_head_71')    # 52x256
        self.conv72 = self.conv_layer(self.conv71, 1, 1, 256, 128, True, 'conv_head_72')    # 52x128
        self.conv73 = self.conv_layer(self.conv72, 3, 1, 128, 256, True, 'conv_head_73')    # 52x256
        self.conv74 = self.conv_layer(self.conv73, 1, 1, 256, 75, False, 'conv_head_74')    # 52x75
        # follow yolo layer mask = 0,1,2

        return self.conv74, self.conv66, self.conv58







