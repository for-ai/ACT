from .registry import register
import tensorflow as tf
import numpy as np
import random

NUM_CLASSES = 15  # i.e. number of sort indices
# These sequence lengths do not include the extra padding we need
# to delay RNN outputs until the entire sequence is seen, which is crucial
# because otherwise our model would have to guess sort order before seeing
# the entire sequence
MIN_SEQUENCE_LENGTH = 2
MAX_SEQUENCE_LENGTH = 15
# Input is padded MAX_SEQUENCE_LENGTH times on the right and likewise for output
# but on the left side
PADDED_SEQUENCE_LENGTH = 30


def generate_batch(batch_size):
  xs = []
  ys = []
  seq_lengths = []
  ms = []
  for _ in range(batch_size):
    x = np.zeros((PADDED_SEQUENCE_LENGTH, 1), np.float32)
    y = np.zeros((PADDED_SEQUENCE_LENGTH,), np.int32)
    target_mask = np.zeros((PADDED_SEQUENCE_LENGTH,), np.int32)

    seq_len = random.randint(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)
    x_random = np.random.normal(0, 1, (seq_len, 1))
    x[:seq_len] = x_random
    # We want this portion to be ignored and one_hot(-1) = all zeros
    y[:seq_len] = -1
    target_mask[seq_len:2 * seq_len] = 1
    y[seq_len:2 * seq_len] = np.argsort(x_random, axis=0).flatten()

    xs.append(x)
    ys.append(y)
    ms.append(target_mask)
    # Note that external seq_len is different because we want the RNN to go
    # over the sequence first in seq_len steps and then emit its outputs
    # one-by-one in another seq_len steps
    seq_lengths.append(2 * seq_len)

  return np.asarray(xs, np.float32), seq_lengths, ys, ms


@register("sort")
def input_fn(data_sources, params, training):

  def _input_fn():
    """
    Returns training inputs and output (y).
    x: 15 element vector having number sequence of random length followed by 0
    y: one-hot encoding representing sorted order of x.
    """
    get_batch = lambda: generate_batch(params.batch_size)
    x, seq_len, y, target_mask = tf.py_func(
        get_batch, [], [tf.float32, tf.int64, tf.int32, tf.int32])
    x.set_shape((params.batch_size, PADDED_SEQUENCE_LENGTH, 1))
    y = tf.one_hot(y, depth=NUM_CLASSES, axis=-1, dtype=tf.int64)
    y.set_shape((params.batch_size, PADDED_SEQUENCE_LENGTH, NUM_CLASSES))
    seq_len.set_shape((params.batch_size,))
    target_mask.set_shape((params.batch_size, PADDED_SEQUENCE_LENGTH))

    return {
        "inputs": x,
        "seq_length": seq_len,
        "difficulty": seq_len,
        "target_mask": target_mask
    }, y

  return _input_fn
