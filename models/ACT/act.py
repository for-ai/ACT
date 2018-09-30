"""Adaptive Computation Time (ACT) Tensorflow implementation


Reference:
- https://arxiv.org/pdf/1603.08983.pdf
- https://github.com/DeNeutoy/act-tensorflow
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..registry import register

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, static_rnn
from tensorboard.plugins.histogram import metadata as histogram_metadata

from ..utils.rnn_utils import sequence_error_rate


class ACTCell(RNNCell):
  """
  A RNN cell implementing Graves' Adaptive Computation Time algorithm
  """

  def __init__(self,
               num_units,
               cell,
               epsilon,
               max_computation,
               batch_size,
               difficulty,
               use_new_ponder_cost=False):

    self.batch_size = batch_size
    self.one_minus_eps = tf.fill([self.batch_size],
                                 tf.constant(1.0 - epsilon, dtype=tf.float32))
    self._num_units = num_units
    self.cell = cell
    self.max_computation = max_computation
    self.remainders = []
    self.iterations = []
    self.use_new_ponder_cost = use_new_ponder_cost
    self.difficulty = difficulty

    if hasattr(self.cell, "_state_is_tuple"):
      self._state_is_tuple = self.cell._state_is_tuple
    else:
      self._state_is_tuple = False

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    if type(self.cell).__name__ == "GRUCell":
      return self._num_units
    else:
      return 2 * self._num_units

  def __call__(self, inputs, state, scope=None):
    if self._state_is_tuple:
      state = tf.concat(state, 1)

    with vs.variable_scope(scope or type(self).__name__):
      # define constants and counters to control while loop for ACTStep
      prob = tf.fill([self.batch_size], tf.constant(0.0, dtype=tf.float32))
      prob_compare = tf.zeros_like(prob, dtype=tf.float32)
      counter = tf.zeros_like(prob, dtype=tf.float32)
      outputs_accumulator = tf.fill([self.batch_size, self.output_size], 0.0)
      states_accumulator = tf.zeros_like(state, dtype=tf.float32)
      batch_mask = tf.fill([self.batch_size], True, name="batch_mask")

      # While loop stops when probability < 1-eps and counter < N are false
      def halting_predicate(batch_mask, prob_compare, prob, counter, state,
                            input, outputs_accumulator, states_accumulator):
        return tf.reduce_any(
            tf.logical_and(
                tf.less(prob_compare, self.one_minus_eps),
                tf.less(counter, self.max_computation)))

      # Do while loop iterations until halting prediction is False
      _, _, halting_prob, iteration, _, _, output, next_state = \
        tf.while_loop(cond=halting_predicate, body=self.act_step,
                      loop_vars=[batch_mask, prob_compare, prob, counter,
                                 state, inputs, outputs_accumulator,
                                 states_accumulator])

    # accumulate remainder and N values
    self.remainders.append(1 - halting_prob)
    self.iterations.append(iteration)

    if self._state_is_tuple:
      next_c, next_h = tf.split(next_state, 2, 1)
      next_state = tf.contrib.rnn.LSTMStateTuple(next_c, next_h)

    return output, next_state

  def calculate_ponder_cost(self, time_penalty, inverse_difficulty):
    """
    time_penalty: scalar
    inverse_difficulty: batch_size x max_len_sequence
    returns tensor of shape batch_size x max_len_sequence which is the ponder cost
    """
    remainders = tf.stack(self.remainders, axis=-1)
    iterations = tf.stack(self.iterations, axis=-1)
    ponder_v1 = remainders + iterations
    if ponder_v1.shape.ndims == 2 and inverse_difficulty.shape.ndims == 3:
      # expand last dimension of ponder_v1 if inverse difficulty and ponder_v1
      # has different dimension
      ponder_v1 = tf.tile(
          tf.expand_dims(ponder_v1, -1), [1, 1, inverse_difficulty.shape[-1]])
    ponder_v2 = inverse_difficulty * ponder_v1
    ponder = ponder_v2 if self.use_new_ponder_cost else ponder_v1
    return time_penalty * ponder

  def act_step(self, batch_mask, prob_compare, prob, counter, state, inputs,
               outputs_accumulator, states_accumulator):
    """
    - generate halting probabilities and accumulate them. Stop
      when the accumulated probs reach a halting value, 1-eps.
    - At each timestep, multiply the prob with the rnn output/state. There is
      a subtlety here regarding the batch_size, as clearly we will have
      examples halting at different points in the batch. This is dealt with
      using logical masks to protect accumulated probabilities, states and
      outputs from a timestep t's contribution if they have already reached
      1 - es at a timestep s < t.
    - On the last timestep for each element in the batch the remainder is
      multiplied with the state/output, having been accumulated over the
      timestep, as this takes into account the epsilon value.
    """
    # set binary flag to 1 when all probs are zero
    binary_flag = tf.cond(
        tf.reduce_all(tf.equal(prob, 0.0)),
        lambda: tf.ones([self.batch_size, 1], tf.float32),
        lambda: tf.zeros([self.batch_size, 1], tf.float32))

    input_with_flag = tf.concat([binary_flag, inputs], 1)

    if self._state_is_tuple:
      (c, h) = tf.split(state, 2, 1)
      state = tf.contrib.rnn.LSTMStateTuple(c, h)

    output, new_state = static_rnn(
        cell=self.cell,
        inputs=[input_with_flag],
        initial_state=state,
        scope=type(self.cell).__name__)

    if self._state_is_tuple:
      new_state = tf.concat(new_state, 1)

    with tf.variable_scope('sigmoid_activation_for_pondering'):
      p = tf.squeeze(
          tf.layers.dense(new_state, 1, activation=tf.sigmoid, use_bias=True),
          axis=1)

    # Multiply by the previous mask as if we stopped before, we don't want to
    # start again if we generate a p less than p_t-1 for a given example.
    new_batch_mask = tf.logical_and(
        tf.less(prob + p, self.one_minus_eps), batch_mask)
    new_float_mask = tf.cast(new_batch_mask, tf.float32)

    # Only increase the prob accumulator for the examples which haven't
    # already passed the threshold. This means that we can just use the final
    # prob value per example to determine the remainder.
    prob += p * new_float_mask

    # This accumulator is used solely in the While loop condition. we multiply
    # by the PREVIOUS batch mask, to capture probabilities that have gone over
    # 1-eps THIS iteration.
    prob_compare += p * tf.cast(batch_mask, tf.float32)

    # Only increase the counter for those probabilities that did not go over
    # 1-eps in this iteration.
    counter += new_float_mask

    # Halting condition (halts, and uses the remainder when this is FALSE):
    # If any batch element still has both a prob < 1 - epsilon AND counter < N
    # continue, using the output probability p.
    counter_condition = tf.less(counter, self.max_computation)

    final_iteration_condition = tf.logical_and(new_batch_mask,
                                               counter_condition)
    use_remainder = tf.expand_dims(1.0 - prob, -1)
    use_probability = tf.expand_dims(p, -1)
    update_weight = tf.where(final_iteration_condition, use_probability,
                             use_remainder)
    float_mask = tf.expand_dims(tf.cast(batch_mask, tf.float32), -1)

    acc_state = (new_state * update_weight * float_mask) + states_accumulator
    acc_output = (output[0] * update_weight * float_mask) + outputs_accumulator

    return [
        new_batch_mask, prob_compare, prob, counter, new_state, inputs,
        acc_output, acc_state
    ]


class ACTModel(object):

  def __init__(self,
               input_data,
               targets,
               difficulty,
               target_mask,
               sequence_length,
               params,
               is_training=False):
    """
    input_data: If non-sequence, then batch_size x feature_size
      otherwise batch_size x max_sequence_length x feature_size
    targets: If non-sequence, then batch_size x num_classes
      otherwise batch_size x max_sequence_length x num_classes
    sequence_length: If non-sequence, then None else tensor with shape batch_size
    """
    self.targets = targets
    self.params = params
    self.batch_size = params.batch_size
    self.hidden_size = params.hidden_size
    self.clip_grad_norm = params.clip_grad_norm
    self.use_lstm = params.use_lstm
    self.difficulty = difficulty
    self.max_difficulty = params.max_difficulty
    self.target_mask = target_mask

    # self.input_data has to be a (length max_sequence_length) list of tensors
    # with shape batch_size x feature_size
    self.input_data = tf.cast(input_data, tf.float32)
    if self.input_data.shape.ndims == 2:
      self.input_data = [self.input_data]
      assert sequence_length is None, 'Non-sequential inputs should leave sequence_length=None'
      sequence_length = tf.constant([1] * self.batch_size)
    elif self.input_data.shape.ndims == 3:
      self.input_data = tf.split(
          self.input_data, num_or_size_splits=self.input_data.shape[1], axis=1)
      self.input_data = [tf.squeeze(t, axis=1) for t in self.input_data]
    else:
      raise Exception('Input has to be of rank 2 or 3')

    # Set up ACT cell and inner rnn-type cell for use inside the ACT cell.
    with tf.variable_scope("rnn"):
      if self.use_lstm:
        inner_cell = BasicLSTMCell(self.hidden_size, state_is_tuple=False)
      else:
        inner_cell = GRUCell(self.hidden_size)

    with tf.variable_scope("ACT"):
      act = ACTCell(
          self.hidden_size,
          inner_cell,
          params.epsilon,
          use_new_ponder_cost=params.use_new_ponder_cost,
          max_computation=params.max_computation,
          batch_size=self.batch_size,
          difficulty=difficulty)

    self.outputs, _ = tf.nn.static_rnn(
        cell=act, inputs=self.input_data, dtype=tf.float32)

    output = tf.stack(self.outputs, axis=1)
    self.logits = tf.layers.dense(output, params.num_classes)

    if params.data == "addition":
      # reshape logits and labels to (batch size, sequence, digits, one hot)
      self.logits = tf.reshape(
          self.logits,
          shape=(params.batch_size, params.max_difficulty,
                 params.num_digits + 1, 10))
      self.targets = tf.reshape(
          self.targets,
          shape=(params.batch_size, params.max_difficulty,
                 params.num_digits + 1, 10))

    self.predictions = tf.nn.softmax(self.logits)
    self.target_mask = tf.cast(self.target_mask, tf.float32)

    ce = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=self.targets, logits=self.logits, dim=-1)
    masked_ce = self.target_mask * ce
    masked_reduced_ce = sparse_mean(masked_ce)

    # Compute the cross entropy based pondering cost multiplier
    avg_ce = tf.Variable(initial_value=0.7, trainable=False)
    avg_ce_decay = 0.85
    avg_ce_update_op = tf.assign(
        avg_ce,
        avg_ce_decay * avg_ce + (1.0 - avg_ce_decay) * masked_reduced_ce)
    with tf.control_dependencies([avg_ce_update_op]):
      inverse_difficulty = safe_div(avg_ce, masked_ce)
      inverse_difficulty /= sparse_mean(inverse_difficulty)
      # ponder_v2 has NaN problem in its backward pass without this
      inverse_difficulty = tf.stop_gradient(inverse_difficulty)

    # Add up loss and retrieve batch-normalised ponder cost: sum N + sum
    # Remainder
    ponder_cost = act.calculate_ponder_cost(
        time_penalty=self.params.ponder_time_penalty,
        inverse_difficulty=inverse_difficulty)

    masked_reduced_ponder_cost = sparse_mean(self.target_mask * ponder_cost)

    self.cost = masked_reduced_ce + masked_reduced_ponder_cost

    if is_training:
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(
          tf.gradients(self.cost, tvars), self.clip_grad_norm)
      optimizer = tf.contrib.estimator.TowerOptimizer(
          tf.train.AdamOptimizer(self.params.learning_rate))
      apply_gradients = optimizer.apply_gradients(zip(grads, tvars))

      gs = tf.train.get_global_step()
      self.train_op = tf.group(apply_gradients, tf.assign_add(gs, 1))

    # Cost metrics
    tf.summary.scalar("ce", masked_reduced_ce)
    tf.summary.scalar("average_inverse_difficulty",
                      sparse_mean(inverse_difficulty * self.target_mask))

    # Pondering metrics
    pondering = tf.stack(act.iterations, axis=-1) + 1

    if params.data == "addition" and pondering.shape.ndims == 2:
      # expand pondering to 3 dimension with repeated last dimension
      pondering = tf.tile(
          tf.expand_dims(pondering, -1), [1, 1, self.target_mask.shape[-1]])

    masked_pondering = self.target_mask * pondering
    dense_pondering = tf.gather_nd(
        masked_pondering, indices=tf.where(tf.not_equal(masked_pondering, 0)))
    tf.summary.scalar("average_pondering", tf.reduce_mean(dense_pondering))
    tf.summary.histogram("pondering", dense_pondering)

    if params.data == "addition":
      avg_pondering = tf.reduce_sum(masked_pondering, axis=[-1, -2]) / \
                      tf.count_nonzero(masked_pondering, axis=[-1, -2],
                                       dtype=tf.float32)
    else:
      avg_pondering = tf.reduce_sum(masked_pondering, axis=-1) / \
                      tf.count_nonzero(masked_pondering, axis=-1,
                                       dtype=tf.float32)

    summary_ponder_metadata = histogram_metadata.create_summary_metadata(
        "difficulty/pondering", "ponder_steps_difficulty")
    summary_ce_metadata = histogram_metadata.create_summary_metadata(
        "difficulty/ce", "ce_steps_difficulty")
    input_difficulty_steps = tf.cast(self.difficulty, tf.float32)
    ponder_steps = tf.cast(avg_pondering, tf.float32)
    ce_steps = tf.cast(masked_reduced_ce, tf.float32)
    ponder_heights = []
    ce_heights = []
    for i in range(self.max_difficulty):
      mask = tf.to_float(tf.equal(self.difficulty, i))
      ponder_avg_steps = tf.cond(
          tf.equal(tf.reduce_sum(mask), 0), lambda: 0.0,
          lambda: tf.reduce_sum(mask * ponder_steps) / tf.reduce_sum(mask))
      ce_avg_steps = tf.cond(
          tf.equal(tf.reduce_sum(mask), 0), lambda: 0.0,
          lambda: tf.reduce_sum(mask * ce_steps) / tf.reduce_sum(mask))
      ponder_heights.append(ponder_avg_steps)
      ce_heights.append(ce_avg_steps)

    ponder_difficulty_steps = tf.transpose(
        tf.stack([
            tf.range(self.max_difficulty, dtype=tf.float32),
            tf.range(self.max_difficulty, dtype=tf.float32) + 1, ponder_heights
        ]))
    ce_difficulty_steps = tf.transpose(
        tf.stack([
            tf.range(self.max_difficulty, dtype=tf.float32),
            tf.range(self.max_difficulty, dtype=tf.float32) + 1, ce_heights
        ]))

    tf.summary.tensor_summary(
        name='ponder_steps_difficulty',
        tensor=ponder_difficulty_steps,
        collections=None,
        summary_metadata=summary_ponder_metadata)

    tf.summary.tensor_summary(
        name='ce_steps_difficulty',
        tensor=ce_difficulty_steps,
        collections=None,
        summary_metadata=summary_ce_metadata)


def sparse_mean(x):
  return tf.reduce_sum(x) / tf.count_nonzero(x, dtype=x.dtype)


def safe_div(x, y):
  return tf.where(tf.less(y, 1e-7), y, x / y)


@register("act")
def get_act(params, lr):
  """Callable model function compatible with Experiment API."""

  params.learning_rate = lr

  def act(features, labels, mode):
    """Basic ACT Model"""

    is_training = mode == tf.contrib.learn.ModeKeys.TRAIN

    with tf.variable_scope("act"):
      act_model = ACTModel(
          input_data=features['inputs'],
          targets=labels,
          sequence_length=features['seq_length'],
          difficulty=features['difficulty'],
          target_mask=features["target_mask"],
          params=params,
          is_training=is_training)

      if params.data == "addition":
        labels = tf.reshape(
            labels,
            shape=(params.batch_size, params.max_difficulty,
                   params.num_digits + 1, 10))

      eval_metric_ops = {
          "acc":
          tf.metrics.accuracy(
              labels=tf.argmax(labels, axis=-1),
              weights=features["target_mask"],
              predictions=tf.argmax(act_model.predictions, axis=-1)),
          "ser":
          sequence_error_rate(
              labels=labels,
              predictions=act_model.predictions,
              target_mask=features["target_mask"])
      }

      if is_training:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=act_model.predictions,
            loss=act_model.cost,
            train_op=act_model.train_op,
            eval_metric_ops=eval_metric_ops)
      else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=act_model.predictions,
            loss=act_model.cost,
            eval_metric_ops=eval_metric_ops)

  return act
