import shutil
import os
import random
import tensorflow as tf
import numpy as np

from .hparams.registry import get_hparams
from .models.registry import _MODELS
from .data.registry import _INPUT_FNS, get_dataset
from .train_utils.lr_schemes import get_lr

tf.set_random_seed(1234)
random.seed(1234)
np.random.seed(1234)

tf.flags.DEFINE_string("model", "ae", "Which model to use.")
tf.flags.DEFINE_string("data", "mnist", "Which data to use.")
tf.flags.DEFINE_string("hparam_sets", "ae", "Which hparams to use.")
tf.flags.DEFINE_string("hparams", "", "Run-specific hparam settings to use.")
tf.flags.DEFINE_string("output_dir", "codebase/tmp/tf_run",
                       "The output directory.")
tf.flags.DEFINE_string("data_dir", "codebase/tmp/data", "The data directory.")
tf.flags.DEFINE_integer("train_steps", 10000,
                        "Number of training steps to perform.")
tf.flags.DEFINE_integer("eval_steps", 100,
                        "Number of evaluation steps to perform.")
tf.flags.DEFINE_integer("eval_every", 1000,
                        "Number of steps between evaluations.")
tf.flags.DEFINE_boolean("overwrite_output", False,
                        "Remove output_dir before running.")
tf.flags.DEFINE_string("train_name", "data-train*",
                       "The train dataset file name.")
tf.flags.DEFINE_string("test_name", "data-eval*", "The test dataset file name.")
tf.flags.DEFINE_boolean("use_new_ponder_cost", False, "Use the new ponder cost.")
FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def _run_locally(train_steps, eval_steps):
  """Run training, evaluation and inference locally.

  Args:
    train_steps: An integer, number of steps to train.
    eval_steps: An integer, number of steps to evaluate.
  """
  hparams = get_hparams(FLAGS.hparam_sets)
  hparams = hparams.parse(FLAGS.hparams)
  hparams.total_steps = FLAGS.train_steps
  hparams.data = FLAGS.data
  hparams.use_new_ponder_cost = FLAGS.use_new_ponder_cost

  output_dir = FLAGS.output_dir
  if os.path.exists(output_dir) and FLAGS.overwrite_output:
    shutil.rmtree(FLAGS.output_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  def model_fn(features, labels, mode):
    lr = get_lr(hparams)
    return _MODELS[FLAGS.model](hparams, lr)(features, labels, mode)

  train_path, eval_path = get_dataset(FLAGS.data_dir, FLAGS.train_name,
                                      FLAGS.test_name)
  train_input_fn = _INPUT_FNS[FLAGS.data](train_path, hparams, training=True)
  eval_input_fn = _INPUT_FNS[FLAGS.data](eval_path, hparams, training=False)

  run_config = tf.contrib.learn.RunConfig(
      save_checkpoints_steps=FLAGS.eval_every)

  estimator = tf.estimator.Estimator(
      model_fn=tf.contrib.estimator.replicate_model_fn(model_fn),
      model_dir=output_dir,
      config=run_config)

  experiment = tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=train_steps,
      eval_steps=eval_steps)

  experiment.train_and_evaluate()


def main(_):
  _run_locally(FLAGS.train_steps, FLAGS.eval_steps)


if __name__ == "__main__":
  tf.app.run()
