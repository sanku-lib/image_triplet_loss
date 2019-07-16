import tensorflow as tf


class TripletLoss:

    def conv_net(self, x, reuse=False):
        with tf.name_scope("model"):
            with tf.variable_scope("conv1") as scope:
                net = tf.contrib.layers.conv2d(x, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv2") as scope:
                net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv3") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv4") as scope:
                net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, 28, [1, 1], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            net = tf.contrib.layers.flatten(net)

        return net


    def triplet_loss(self, model_anchor, model_positive, model_negative, margin):
        distance1 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_positive, 2), 1, keepdims=True))
        distance2 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_negative, 2), 1, keepdims=True))
        return tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0))