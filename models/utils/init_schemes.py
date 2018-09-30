import tensorflow as tf

_INIT = dict()


def register(name):

  def add_to_dict(fn):
    global _INIT
    _INIT[name] = fn
    return fn

  return add_to_dict


def get_init(params):
  return _INIT[params.init_scheme](params)


@register("normal")
def normal(params):
  return tf.random_normal_initializer()


@register("constant")
def constant(params):
  return tf.constant_initializer(0.1, tf.int32)


@register("uniform_unit_scaling")
def uniform_unit_scaling(params):
  return tf.uniform_unit_scaling_initializer()
