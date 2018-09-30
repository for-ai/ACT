import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, RNNCell, static_rnn, GRUCell
from ..registry import register
from ..utils.rnn_utils import sequence_error_rate


@register("rnn")
def get_rnn(params, lr):
  """Callable model function compatible with Experiment API.
  Args:
  params: a HParams object containing values for fields:
  input size: The size of the input sequence
  output size: The size of the output
  rnn_hidden: Number of hidden layers size
  use_lstm: A binary variable to use lstm or not
  lr: learning rate variable
    """

  def rnn(features, labels, mode):
    """The basic lstm template.
    Args:
    features: a dict containing key "inputs", "seq_length"
    mode: training, evaluation or infer
    """
    with tf.variable_scope("rnn"):
      is_training = mode == tf.contrib.learn.ModeKeys.TRAIN
      x = tf.cast(features["inputs"], tf.float32)
      sequence_length = features["seq_length"]
      target_mask = features["target_mask"]

      if x.shape.ndims == 2:
        # Make non-sequential data compatible
        assert labels.shape.ndims == 2, 'Inputs and targets have different rank'
        x, labels = tf.expand_dims(x, axis=1), tf.expand_dims(labels, axis=1)

      if params.use_lstm:
        cell = BasicLSTMCell(params.hidden_size)
      else:
        cell = GRUCell(params.hidden_size)

      outputs, state = tf.nn.dynamic_rnn(
          cell,
          x,
          sequence_length=features["seq_length"],
          dtype=tf.float32,
          time_major=False)

      logits = tf.layers.dense(outputs, params.num_classes)

      if params.data == "addition":
        # reshape logits and labels to (batch size, sequence, digits, one hot)
        logits = tf.reshape(logits,
                            shape=(params.batch_size, params.max_difficulty,
                                   params.num_digits + 1, 10))
        labels = tf.reshape(labels, shape=(params.batch_size,
                                           params.max_difficulty,
                                           params.num_digits + 1, 10))
      ce = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, dim=-1)
      mask_ce = ce * tf.cast(target_mask, tf.float32)
      cost = tf.reduce_mean(mask_ce)

      optimizer = tf.train.AdamOptimizer(params.learning_rate).minimize(
          cost, global_step=tf.train.get_global_step())
      predictions = tf.nn.softmax(logits)

      gs = tf.contrib.framework.get_global_step()
      train_op = tf.group(optimizer, tf.assign_add(gs, 1))

      # These two are different for sequences but same otherwise
      eval_metric_ops = {
        "acc":
          tf.metrics.accuracy(
            labels=tf.argmax(labels, -1),
            weights=target_mask,
            predictions=tf.argmax(predictions, -1)),
        "ser":
          sequence_error_rate(
            labels=labels,
            predictions=predictions,
            target_mask=target_mask)
      }

      if is_training:
        return tf.estimator.EstimatorSpec(
          mode,
          predictions=predictions,
          loss=cost,
          train_op=train_op)
      else:
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=cost,
          eval_metric_ops=eval_metric_ops)

  return rnn
