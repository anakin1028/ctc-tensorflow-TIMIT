"""
Module to train the CTC model
"""

import tensorflow as tf
import numpy as np
# for getting the feature extraction
import featext
# for generating the batch data
import gen_batch

# 13MFCCS and their first order differential
NUM_FEATURES = 26
# 100 LSTM cells in a block
NUM_UNITS = 100
# 61 phonemes + 1 NULL
NUM_CLASSES = 62
# Hyper parameters
NUM_EPOCHS = 450
# how deep is the model
NUM_LAYERS = 1
BATCH_SIZE = 30
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
# number of training data: 4620
# batch_size: 50
NUM_EXAMPLES = 3696
NUM_BATCHES = 124
# output model
OUTPUT_PATH = "./output/inference_model"

def pad_zeros_for_inputs(batch_data):
    """
    Input looks like as
    [
        [[t1],[t2],[t3]],          # instance1
        [[t1], [t2], ],            # instance2
    ]
    Output should be
    [
        [[t1],[t2],[t3]],          # instance1
        [[t1], [t2], [zeros]],     # instance2
    ],
    List of timestamps [3,         # instance 1
                        2]         # instance 2
    """
    shape_list = [data.shape[0] for data in batch_data]
    max_timestamps_idx = np.argmax(shape_list)
    max_timestamps = shape_list[max_timestamps_idx]
    # times for each instance
    batch_timestamps = []
    # padding the input data
    paddatas = []
    for data in batch_data:
        # shape[0] is number of timestamps in the data
        timestamps = data.shape[0]
        pad_rows = max_timestamps - timestamps
        # pad row only
        paddata = np.pad(data, [(0, pad_rows), (0, 0)], mode='constant')
        batch_timestamps.append(timestamps)
        paddatas.append(paddata)
    return paddatas, batch_timestamps

def get_train_inputs():
    """
    Get the training data in the pickle file
    """
    phoneme_objs = featext.get_train_data()
    all_inputs = []
    # number of output sequences
    all_outputs = []
    # normalize the inputs
    for phoneme_obj in phoneme_objs:
        features = phoneme_obj.features
        normalize_features = (features - np.mean(features))/np.std(features)
        all_inputs.append(normalize_features)
        all_outputs.append(phoneme_obj.get_phone_stream())
    inputs, timestamp_seq = pad_zeros_for_inputs(all_inputs)
    return inputs, timestamp_seq, all_outputs

def transform_single_example(features, phnfile):
    """
    Used for validation
    """
    normalize_features = (features - np.mean(features))/np.std(features)
    phones = []
    with open(phnfile, 'r') as fh:
        for line in fh:
            phone = line.strip().split(" ")[2]
            phones.append(featext.PHONE_IDX_DICT[phone])
    inputs, timestamp_seq = pad_zeros_for_inputs([normalize_features])
    return inputs, timestamp_seq, [phones]

def get_validation_input(audio_path, phnfile):
    """
    path for extracting the mfcc and running validation
    """
    features = featext.get_audio_feature(audio_path)
    return transform_single_example(features, phnfile)

def train():
    """
    Build and train the model
    """
    train_inputs, timestamp_seq, outputs = get_train_inputs()
    gen_batch_obj = gen_batch.GenBatchData(train_inputs, timestamp_seq,
                                           outputs, BATCH_SIZE)
    graph = tf.Graph()
    with graph.as_default():
        # input mfcc datas
        inputs = tf.placeholder(tf.float32, [None, None, NUM_FEATURES], name="inputs")
        # output phonemes
        targets = tf.sparse_placeholder(tf.int32, name="outputs")
        seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
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
        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.MomentumOptimizer(LEARNING_RATE,
                                               MOMENTUM).minimize(cost)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for curr_epoch in range(NUM_EPOCHS):
            train_cost = train_ler = 0
            for i in range(NUM_BATCHES):
                b_inputs, b_seqlen, b_outputs = gen_batch_obj.get_batch_data()
                feed = {
                    inputs: b_inputs,
                    targets: b_outputs,
                    seq_len: b_seqlen
                }
                batch_cost, _ = session.run([cost, optimizer], feed)
                batch_ler = session.run(ler, feed_dict=feed)*BATCH_SIZE
                train_cost += batch_cost * BATCH_SIZE
                train_ler += batch_ler
                print("current epoch:{} current batch:{} batch_cost: {} batch_ler: {}".format(
                    curr_epoch, i, batch_cost, batch_ler))
            train_cost /= NUM_EXAMPLES
            train_ler /= NUM_EXAMPLES
            print("epoch {} cost {} ler {}".format(curr_epoch, train_cost,
                                                   train_ler))
        saver = tf.train.Saver()
        saver.save(session, OUTPUT_PATH)

def validate(audiopath, phnfile):
    """
    running Validation
    """
    train_inputs, timestamp_seq, originals = get_validation_input(audiopath, phnfile)
    gen_batch_obj = gen_batch.GenBatchData(train_inputs, timestamp_seq,
                                           originals, 1)
    # pick one batch
    b_inputs, b_seqlen, b_outputs = gen_batch_obj.get_batch_data()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("./output/inference_model.meta")
        saver.restore(sess, tf.train.latest_checkpoint("./output"))
        graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name("inputs:0")
        # extract indices/values/shapes for the sparse tensor
        outputs_indices = graph.get_tensor_by_name("outputs/indices:0")
        outputs_values = graph.get_tensor_by_name("outputs/values:0")
        outputs_shapes = graph.get_tensor_by_name("outputs/shape:0")
        seq_len = graph.get_tensor_by_name("seq_len:0")
        logit_operation = graph.get_tensor_by_name("output_logits:0")
        feed = {
            inputs: b_inputs,
            outputs_indices: b_outputs[0],
            outputs_values: b_outputs[1],
            outputs_shapes: b_outputs[2],
            seq_len: b_seqlen
        }
        logits = sess.run(logit_operation, feed)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
        model_decode = sess.run(decoded[0], feed_dict=feed)
        phoneme_decoded = ' '.join([featext.IDX2PHONE[x] for x in model_decode[1]])
        original_decoded = ' '.join([featext.IDX2PHONE[x] for x in originals[0]])
        print("model decoded    {}".format(phoneme_decoded))
        print("original decoded {}".format(original_decoded))
