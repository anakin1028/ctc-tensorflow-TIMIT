"""
Module to generate the batch data
"""

import numpy as np
import tensorflow as tf

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    return indices, values, shape

class GenBatchData:
    """
    Class for generating the batch data in training and testing
    """
    def __init__(self, inputs, timestamp_seq, all_outputs,
                 batch_size):
        """
        construct the object
        """
        self.inputs = inputs
        self.timestamp_seq = timestamp_seq
        self.all_outputs = all_outputs
        self.batch_size = batch_size
        self.current_idx = 0
        self.number_of_inputs = len(self.inputs)

    def get_batch_seq(self, input_seq):
        """
        Generate the sparse output
        """
        current_idx = self.current_idx
        if self.number_of_inputs - current_idx >= self.batch_size:
            batch_outputs = input_seq[
                current_idx:current_idx+self.batch_size]
        else:
            tmp = self.number_of_inputs - current_idx
            batch_outputs = input_seq[current_idx:]
            current_idx = self.batch_size - tmp
            batch_outputs += input_seq[:current_idx]
        return batch_outputs

    def get_batch_data(self):
        """
        Generate the batch data for input, timestamp sequence, and output
        """
        inputs = self.get_batch_seq(self.inputs)
        timestamps = self.get_batch_seq(self.timestamp_seq)
        outputs = self.get_batch_seq(self.all_outputs)
        # update current index
        if self.number_of_inputs - self.current_idx >= self.batch_size:
            self.current_idx = (self.current_idx + self.batch_size) % self.number_of_inputs
        else:
            tmp = self.number_of_inputs - self.current_idx
            self.current_idx = self.batch_size - tmp
        return np.asarray(inputs), np.asarray(timestamps), sparse_tuple_from(outputs)

if __name__ == "__main__":
    inputs = [[1,2,3],[4,5,6],[7,8,9]]
    times = [98, 99, 100]
    outputs = [[10],[14],[23]]
    batch_size = 2
    batch_data = GenBatchData(inputs, times, outputs, batch_size)
    for i in range(10):
        print(batch_data.get_batch_data())
