import tensorflow as tf


def sequence_error_rate(labels, predictions, target_mask):
  """ Return the sequence error rate of labels and predictions
  The fraction of examples where any mistakes were made in the complete
  output sequence

  NOTE: predictions (softmax) and labels (one-hot) must share the same
  shape of batch_size x max_seq_len x output_size or batch_size output_size
  i.e. both sequence and non-sequence inputs are accepted
  """
  if labels.shape.ndims != predictions.shape.ndims:
    # remove size 1 dimensions in prediction
    predictions = tf.squeeze(predictions)

  assert labels.shape.ndims == predictions.shape.ndims, \
    'labels and predictions have different rank'
  predictions = tf.argmax(predictions, axis=-1)
  labels = tf.argmax(labels, axis=-1)
  is_correct_per_sample = tf.equal(predictions, labels)
  if predictions.shape.ndims > 1:
    is_correct_per_sample = tf.logical_or(
        is_correct_per_sample, tf.logical_not(tf.cast(target_mask, tf.bool)))
    is_correct_per_sample = tf.reduce_all(is_correct_per_sample, axis=-1)
  mean, update_op = tf.metrics.mean(tf.cast(is_correct_per_sample, tf.float32))
  return 1 - mean, update_op
