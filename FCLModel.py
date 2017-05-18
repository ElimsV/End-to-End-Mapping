import tensorflow as tf


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def fc_layer(input,  # The previous layer
             num_inputs,  # Num. inputs from prev. layer
             num_outputs,  # Num. outputs
             drop_rate,  # Probability for drop out
             use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer
    layer = tf.matmul(input, weights) + biases

    # Activation function
    if use_relu:
        layer = tf.nn.relu(layer)

    if drop_rate > 0:
        layer = tf.nn.dropout(layer, drop_rate)

    return layer