import tensorflow as tf

import numpy as np
conv_bn = ['bias', 'gamma', 'mean', 'variance', 'conv_weights']

class Darknet53:
    def __init__(self, darknet53_npz_path=None, trainable=True, scratch=False):
        """
        :param darknet53_npz_path:
        :param trainable:   python
        :param phase_train: tensor
        """
        if darknet53_npz_path is not None:
            self.data_dict = np.load(darknet53_npz_path)
            self.keys = self.data_dict.keys()
        self.trainable = trainable
        self.avg_var = []
        self.scratch = scratch

    def get_var(self, initial_value, name, var_name, var_trainable=True):
        """

        :param initial_value:
        :param name:
        :param var_name:
        :param trainable:  moving average not trainable
        :return:
        """
        if self.scratch:
            value = initial_value
        elif self.data_dict is not None and name in self.keys:
            idx = conv_bn.index(var_name)
            value = self.data_dict[name][idx] if idx < 4 else self.data_dict[name][idx][0]
        else:
            raise ValueError('From scratch train feature extractor or provide complete weights')

        if self.trainable and var_trainable:
            var = tf.Variable(value, name=var_name)
        elif self.trainable:
            var = tf.Variable(value, name=var_name, trainable=False)
            self.avg_var.append(var)
        else:
            var = tf.const(value, dtype=tf.float32, name=var_name)
        return var

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 'conv_weights')

        return filters

    def get_conv_bn_var(self, out_channels, name):
        initial_value = tf.truncated_normal([out_channels], .0, .001)
        beta = self.get_var(initial_value, name, 'bias')

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        gamma = self.get_var(initial_value, name, 'gamma')

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        mean = self.get_var(initial_value, name, 'mean', var_trainable=False)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        variance = self.get_var(initial_value, name, 'variance', var_trainable=False)
        return beta, gamma, mean, variance

    def conv_layer(self, bottom, size, stride, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt = self.get_conv_var(size, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
            beta, gamma, moving_mean, moving_variance = self.get_conv_bn_var(out_channels, name)
            # TODO: How to initialize moving_mean & moving_var with darknet data
            def mean_var_with_update():
                batch_mean, batch_var = tf.nn.moments(conv, [0, 1, 2], name='moments')
                train_mean = tf.assign(moving_mean,
                                       moving_mean*self.decay_bn + batch_mean*(1 - self.decay_bn))
                train_var = tf.assign(moving_variance,
                                      moving_variance * self.decay_bn + moving_variance * (1 - self.decay_bn))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(self.phase_train,
                                mean_var_with_update,
                                lambda: (moving_mean, moving_variance))
            normed = tf.nn.batch_normalization(conv, mean, var, beta, gamma, 1e-8)
            activation = tf.nn.leaky_relu(normed, alpha=0.1)
        return activation

    def build(self, img, istraining, decay_bn=0.99):
        self.phase_train = istraining
        self.decay_bn = decay_bn
        self.conv0 = self.conv_layer(bottom=img, size=3, stride=1, in_channels=3,   # 416x3
                                     out_channels=32, name='conv_0')                # 416x32
        self.conv1 = self.conv_layer(bottom=self.conv0, size=3, stride=2, in_channels=32,
                                     out_channels=64, name='conv_1')                # 208x64
        self.conv2 = self.conv_layer(bottom=self.conv1, size=1, stride=1, in_channels=64,
                                     out_channels=32, name='conv_2')                # 208x32
        self.conv3 = self.conv_layer(bottom=self.conv2, size=3, stride=1, in_channels=32,
                                     out_channels=64, name='conv_3')                # 208x64
        self.res0 = self.conv3 + self.conv1                                         # 208x64
        self.conv4 = self.conv_layer(bottom=self.res0, size=3, stride=2, in_channels=64,
                                     out_channels=128, name='conv_4')               # 104x128
        self.conv5 = self.conv_layer(bottom=self.conv4, size=1, stride=1, in_channels=128,
                                     out_channels=64, name='conv_5')                # 104x64
        self.conv6 = self.conv_layer(bottom=self.conv5, size=3, stride=1, in_channels=64,
                                     out_channels=128, name='conv_6')               # 104x128
        self.res1 = self.conv6 + self.conv4     # 128                               # 104x128
        self.conv7 = self.conv_layer(bottom=self.res1, size=1, stride=1, in_channels=128,
                                     out_channels=64, name='conv_7')                # 104x64
        self.conv8 = self.conv_layer(bottom=self.conv7, size=3, stride=1, in_channels=64,
                                     out_channels=128, name='conv_8')               # 104x128
        self.res2 = self.conv8 + self.res1      # 128                               # 104x128
        self.conv9 = self.conv_layer(bottom=self.res2, size=3, stride=2, in_channels=128,
                                     out_channels=256, name='conv_9')               # 52x256
        self.conv10 = self.conv_layer(bottom=self.conv9, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_10')             # 52x128
        self.conv11 = self.conv_layer(bottom=self.conv10, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_11')             # 52x256
        self.res3 = self.conv11 + self.conv9                                        # 52x256
        self.conv12 = self.conv_layer(bottom=self.res3, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_12')             # 52x128
        self.conv13 = self.conv_layer(bottom=self.conv12, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_13')             # 52x256
        self.res4 = self.conv13 + self.res3                                         # 52x256
        self.conv14 = self.conv_layer(bottom=self.res4, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_14')             # 52x128
        self.conv15 = self.conv_layer(bottom=self.conv14, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_15')             # 52x256
        self.res5 = self.conv15 + self.res4                                         # 52x256
        self.conv16 = self.conv_layer(bottom=self.res5, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_16')             # 52x128
        self.conv17 = self.conv_layer(bottom=self.conv16, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_17')             # 52x256
        self.res6 = self.conv17 + self.res5                                         # 52x256
        self.conv18 = self.conv_layer(bottom=self.res6, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_18')             # 52x128
        self.conv19 = self.conv_layer(bottom=self.conv18, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_19')             # 52x256
        self.res7 = self.conv19 + self.res6                                         # 52x256
        self.conv20 = self.conv_layer(bottom=self.res7, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_20')             # 52x128
        self.conv21 = self.conv_layer(bottom=self.conv20, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_21')             # 52x256
        self.res8 = self.conv21 + self.res7                                         # 52x256
        self.conv22 = self.conv_layer(bottom=self.res8, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_22')             # 52x128
        self.conv23 = self.conv_layer(bottom=self.conv22, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_23')             # 52x256
        self.res9 = self.conv23 + self.res8                                         # 52x256
        self.conv24 = self.conv_layer(bottom=self.res9, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_24')             # 52x128
        self.conv25 = self.conv_layer(bottom=self.conv24, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_25')             # 52x256
        self.res10 = self.conv25 + self.res9                                        # 52x256
        self.conv26 = self.conv_layer(bottom=self.res10, size=3, stride=2, in_channels=256,
                                      out_channels=512, name='conv_26')             # 26x512
        self.conv27 = self.conv_layer(bottom=self.conv26, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_27')             # 26x256
        self.conv28 = self.conv_layer(bottom=self.conv27, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_28')             # 26x512
        self.res11 = self.conv28 + self.conv26                                      # 26x512
        self.conv29 = self.conv_layer(bottom=self.res11, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_29')             # 26x256
        self.conv30 = self.conv_layer(bottom=self.conv29, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_30')             # 26x512
        self.res12 = self.conv30 + self.res11                                       # 26x512
        self.conv31 = self.conv_layer(bottom=self.res12, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_31')             # 26x256
        self.conv32 = self.conv_layer(bottom=self.conv31, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_32')             # 26x512
        self.res13 = self.conv32 + self.res12                                       # 26x512
        self.conv33 = self.conv_layer(bottom=self.res13, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_33')             # 26x256
        self.conv34 = self.conv_layer(bottom=self.conv33, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_34')             # 26x512
        self.res14 = self.conv34 + self.res13                                       # 26x512
        self.conv35 = self.conv_layer(bottom=self.res14, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_35')             # 26x256
        self.conv36 = self.conv_layer(bottom=self.conv35, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_36')             # 26x512
        self.res15 = self.conv36 + self.res14                                       # 26x512
        self.conv37 = self.conv_layer(bottom=self.res15, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_37')             # 26x256
        self.conv38 = self.conv_layer(bottom=self.conv37, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_38')             # 26x512
        self.res16 = self.conv38 + self.res15                                       # 26x512
        self.conv39 = self.conv_layer(bottom=self.res16, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_39')             # 26x256
        self.conv40 = self.conv_layer(bottom=self.conv39, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_40')             # 26x512
        self.res17 = self.conv40 + self.res16                                       # 26x512
        self.conv41 = self.conv_layer(bottom=self.res17, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_41')             # 26x256
        self.conv42 = self.conv_layer(bottom=self.conv41, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_42')             # 26x512
        self.res18 = self.conv42 + self.res17                                       # 26x512
        self.conv43 = self.conv_layer(bottom=self.res18, size=3, stride=2, in_channels=512,
                                      out_channels=1024, name='conv_43')            # 13x1024
        self.conv44 = self.conv_layer(bottom=self.conv43, size=1, stride=1, in_channels=1024,
                                      out_channels=512, name='conv_44')             # 13x512
        self.conv45 = self.conv_layer(bottom=self.conv44, size=3, stride=1, in_channels=512,
                                      out_channels=1024, name='conv_45')            # 13x1024
        self.res19 = self.conv45 + self.conv43                                      # 13x1024
        self.conv46 = self.conv_layer(bottom=self.res19, size=1, stride=1, in_channels=1024,
                                      out_channels=512, name='conv_46')             # 13x512
        self.conv47 = self.conv_layer(bottom=self.conv44, size=3, stride=1, in_channels=512,
                                      out_channels=1024, name='conv_47')            # 13x1024
        self.res20 = self.conv47 + self.res19                                       # 13x1024
        self.conv48 = self.conv_layer(bottom=self.res20, size=1, stride=1, in_channels=1024,
                                      out_channels=512, name='conv_48')             # 13x512
        self.conv49 = self.conv_layer(bottom=self.conv48, size=3, stride=1, in_channels=512,
                                      out_channels=1024, name='conv_49')            # 13x1024
        self.res21 = self.conv49 + self.res20                                       # 13x1024
        self.conv50 = self.conv_layer(bottom=self.res21, size=1, stride=1, in_channels=1024,
                                      out_channels=512, name='conv_50')             # 13x512
        self.conv51 = self.conv_layer(bottom=self.conv50, size=3, stride=1, in_channels=512,
                                      out_channels=1024, name='conv_51')            # 13x1024
        self.res23 = self.conv51 + self.res21                                       # 13x1024
        return self.res23

















