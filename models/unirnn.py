"""
Define the models
"""

import tensorflow as tf
# 100 LSTM cells in a block
NUM_UNITS = 100
# 61 phonemes + 1 NULL
NUM_CLASSES = 62

def inference(inputs, seq_len):
    """
    Create the logits for decoding or training
    """
    cell = tf.contrib.rnn.LSTMCell(NUM_UNITS)
    outputs, states = tf.nn.dynamic_rnn(cell, inputs,
                                        sequence_length=seq_len,
                                        dtype=tf.float32)
    # the input shape
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]
    outputs = tf.reshape(outputs, [-1, NUM_UNITS])
    W = tf.Variable(tf.truncated_normal([NUM_UNITS, NUM_CLASSES],
                                        stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]))
    logits = tf.matmul(outputs, W) + b
    # reshape to original shape
    logits = tf.reshape(logits, [batch_s, -1, NUM_CLASSES])
    # time major
    logits = tf.transpose(logits, (1, 0, 2), name="output_logits")
    return logits
