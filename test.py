import tensorflow as tf


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


a = tf.random_normal_initializer()
welcome = tf.constant('Welcome to TensorFlow world!')

with tf.variable_scope("Linear"):
    matrix = tf.get_variable("Matrix", [10, 20], tf.float32,
                             tf.random_normal_initializer(stddev=1))

with tf.Session() as sess:
    print(sess.run(matrix))
    print(sess.run(welcome))
    print(sess.run(a))
