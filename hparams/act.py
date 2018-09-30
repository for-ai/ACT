import tensorflow as tf

from .registry import register
from .defaults import default


@register("act_parity")
def act_parity():
  act = default()

  act.num_classes = 2
  act.batch_size = 128
  act.lr_scheme = "constant"
  act.label_smoothing = 0
  act.learning_rate = 10e-4

  act.hidden_size = 128
  act.use_lstm = False

  act.max_computation = 100
  act.max_difficulty = 64
  act.epsilon = 0.01
  act.ponder_time_penalty = 0.001
  act.use_new_ponder_cost = False

  return act


@register("act_sort")
def act_sort():
  act = default()

  act.num_classes = 15
  act.batch_size = 16
  act.lr_scheme = "constant"
  act.label_smoothing = 0
  act.learning_rate = 10e-4

  act.hidden_size = 512
  act.use_lstm = False

  act.max_computation = 100
  act.max_difficulty = 30
  act.epsilon = 0.01
  act.ponder_time_penalty = 0.001
  act.use_new_ponder_cost = False

  return act


@register("act_addition")
def act_addition():
  act = default()

  act.num_classes = 60
  act.batch_size = 64
  act.lr_scheme = "constant"
  act.label_smoothing = 0
  act.learning_rate = 10e-4

  act.hidden_size = 512
  act.use_lstm = True

  act.max_computation = 100
  act.max_difficulty = 5
  act.max_sequence_length = 5
  act.num_digits = 5
  act.epsilon = 0.01
  act.ponder_time_penalty = 0.001
  act.use_new_ponder_cost = False

  return act
