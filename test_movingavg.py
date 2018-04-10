import tensorflow as tf
batch_mean = tf.constant([1], dtype=tf.float32)


moving_mean = tf.Variable([10], trainable=False, dtype=tf.float32)

decay_bn = tf.constant(0.9, dtype=tf.float32)
phase_train = tf.placeholder(tf.bool, shape=[])

def mean_var_with_update():
    train_mean = tf.assign(moving_mean,
                           moving_mean * decay_bn + batch_mean * (1 - decay_bn))
    with tf.control_dependencies([train_mean]):
        return tf.identity(batch_mean)

mean = tf.cond(phase_train,
                    mean_var_with_update,
                    lambda: tf.identity(moving_mean))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([mean, moving_mean], feed_dict={phase_train: True}))
print(sess.run([mean, moving_mean], feed_dict={phase_train: False}))
print(sess.run([mean, moving_mean], feed_dict={phase_train: False}))