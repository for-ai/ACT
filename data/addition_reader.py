import tensorflow as tf
import random
import numpy as np
from .registry import register


def generate_batch(params):
  features = []
  labels = []
  sequence_lengths = []
  target_mask = []

  # helper to mask total_list
  total_mask = [-1] * (params.num_digits + 1)

  for _ in range(params.batch_size):
    feature = []
    label = []
    total = 0
    mask = []

    sequence_length = random.randint(1, params.max_difficulty)
    sequence_lengths.append(sequence_length)

    # generate one number
    for i in range(sequence_length):
      num_digits = random.randint(1, params.num_digits)
      value = []
      for _ in range(num_digits):
        value.append(random.randint(0, 9))
      # remove zero if value start with zero
      while value and value[0] == 0:
        value.pop(0)
      # add to total if value is not zero
      if value:
        total += int(''.join(map(str, value)))
      # convert int to list of int
      total_list = list(map(int, str(total)))
      # pad value with -1
      value += [-1] * (params.num_digits - len(value))
      # pad digits beyond the end of target number with 11
      total_list += [-1] * (params.num_digits + 1 - len(total_list))
      # mask number of digits
      mask.append((1 - np.equal(total_list, total_mask).astype(int)).tolist())
      feature.append(value)
      label.append(total_list)

    # pad samples, targets to sequence length
    sequence_offset = params.max_difficulty - sequence_length
    feature += [[-1] * params.num_digits] * sequence_offset
    label += [[-1] * (params.num_digits + 1)] * sequence_offset
    mask += [[0] * (params.num_digits + 1)] * sequence_offset

    assert len(feature) == len(label) == len(mask)

    features.append(feature)
    labels.append(label)
    target_mask.append(mask)

  return features, sequence_lengths, target_mask, labels


@register("addition")
def input_fn(data_sources, params, training):

  def _input_fn():
    """ Generate batch_size number of addition samples

    pad x with -1 and y with 11 to match sequence length and number of digits

    y has 11 classes, the 11th class represent number is complete

    Returns:
      x: shape=(batch_size, max_difficulty, num_digits * 10),
        randomly generated integer
      seq_length: shape(batch_size,). sequence length for each input
      y: shape=(batch_size, max_difficulty, num_digits + 1 * 11),
        sum of x until the current index
    """
    get_batch = lambda: generate_batch(params)

    x, seq_length, target_mask, y = \
      tf.py_func(get_batch, [], [tf.int64, tf.int64, tf.int64, tf.int64])

    x = tf.reshape(
        tf.one_hot(x, depth=10),
        shape=(params.batch_size, params.max_difficulty,
               params.num_digits * 10))

    y = tf.reshape(
        tf.one_hot(y, depth=10),
        shape=(params.batch_size, params.max_difficulty, params.num_classes))

    seq_length.set_shape(shape=(params.batch_size,))

    target_mask.set_shape(
        shape=(params.batch_size, params.max_difficulty, params.num_digits + 1))

    return {
        "inputs": x,
        "seq_length": seq_length,
        "difficulty": seq_length,
        "target_mask": target_mask
    }, y

  return _input_fn
