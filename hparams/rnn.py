import tensorflow as tf

from .registry import register
from .defaults import default


@register("rnn_parity")
def rnn_parity():
  rnn = default()
  rnn.input_size = None
  rnn.batch_size = 16
  rnn.learning_rate = 10e-4
  rnn.num_classes = 2
  rnn.max_sequence_length = 64
  rnn.output_size = None
  rnn.hidden_size = 128
  rnn.use_lstm = True
  return rnn


@register("rnn_sort")
def rnn_sort():
  rnn = default()
  rnn.input_size = None
  rnn.batch_size = 16
  rnn.learning_rate = 10e-4
  rnn.num_classes = 15
  rnn.max_sequence_length = 30
  rnn.output_size = None
  rnn.hidden_size = 512
  rnn.use_lstm = True
  return rnn


@register("rnn_addition")
def rnn_addition():
  rnn = default()
  rnn.input_size = None
  rnn.batch_size = 64
  rnn.learning_rate = 10e-4
  rnn.num_classes = 60
  rnn.output_size = None
  rnn.hidden_size = 512
  rnn.use_lstm = True
  rnn.max_difficulty = 5
  rnn.num_digits = 5
  rnn.max_sequence_length = 5
  rnn.reshape = True
  return rnn
