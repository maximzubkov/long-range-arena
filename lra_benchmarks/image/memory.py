# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main training script for the image classification task."""
import functools
import itertools
import os
from os.path import exists, join

import jax
import jax.nn
import jax.numpy as jnp
import tensorflow.compat.v2 as tf
from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import nn
from flax import optim
from flax.training import common_utils
from jax import random
from ml_collections import config_flags

from lra_benchmarks.image import task_registry
from lra_benchmarks.models.transformer import transformer

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_string('task_name', default='mnist', help='Name of the task')
flags.DEFINE_bool(
    'eval_only', default=False, help='Run the evaluation on the test data.')
flags.DEFINE_string(
    'results', default=None,
    help='Name of the JSON file to store the results.')


def create_model(key, flax_module, input_shape, model_kwargs):
  """Creates and initializes the model."""

  @functools.partial(jax.jit, backend='cpu')
  def _create_model(key):
    module = flax_module.partial(**model_kwargs)
    with nn.stateful() as init_state:
      with nn.stochastic(key):
        _, initial_params = module.init_by_shape(key,
                                                 [(input_shape, jnp.float32)])
        model = nn.Model(module, initial_params)
    return model, init_state

  return _create_model(key)


def create_optimizer(model, learning_rate, weight_decay):
  optimizer_def = optim.Adam(
      learning_rate, beta1=0.9, beta2=0.98, eps=1e-9, weight_decay=weight_decay)
  optimizer = optimizer_def.create(model)
  return optimizer


def get_model(init_rng, input_shape, model_type, model_kwargs):
  """Create and initialize the model.

  Args:
    init_rng: float; Jax PRNG key.
    input_shape: tuple; Tuple indicating input shape.
    model_type: str; Type of Transformer model to create.
    model_kwargs: keyword argument to the model.

  Returns:
    Initialized model.
  """
  if model_type == 'transformer':
    return create_model(init_rng, transformer.TransformerEncoder, input_shape,
                        model_kwargs)
  else:
    raise ValueError('Model type not supported')


def eval_step(model, state, batch, num_classes, flatten_input=True):
  eval_keys = ['inputs', 'targets']
  (inputs, targets) = [batch.get(k, None) for k in eval_keys]
  if flatten_input:
    inputs = inputs.reshape(inputs.shape[0], -1)
  if jax.tree_leaves(state):
    state = jax.lax.pmean(state, 'batch')
  with nn.stateful(state, mutable=False):
    logits = model(inputs, train=False)
  return logits


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  config = FLAGS.config
  logging.info('===========Config Dict============')
  logging.info(config)
  batch_size = config.batch_size
  learning_rate = config.learning_rate
  random_seed = config.random_seed
  model_type = config.model_type

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  logging.info('Training on %s', FLAGS.task_name)

  if model_type in ['wideresnet', 'resnet', 'simple_cnn']:
    normalize = True
  else:  # transformer-based models
    normalize = False
  (train_ds, eval_ds, test_ds, num_classes, vocab_size,
   input_shape) = task_registry.TASK_DATA_DICT[FLAGS.task_name](
       n_devices=jax.local_device_count(),
       batch_size=batch_size,
       normalize=normalize)
  train_iter = iter(train_ds)
  model_kwargs = (
      config.model_kwargs.to_dict() if 'model_kwargs' in config else {})
  flatten_input = True

  if model_type in ['wideresnet', 'resnet', 'simple_cnn']:
    model_kwargs.update({
        'num_classes': num_classes,
    })
    flatten_input = False

  else:  # transformer models
    # we will flatten the input
    bs, h, w, c = input_shape
    assert c == 1
    input_shape = (bs, h * w * c)
    model_kwargs.update({
        'vocab_size': vocab_size,
        'max_len': input_shape[1],
        'classifier': True,
        'num_classes': num_classes,
    })

  model_kwargs.update(config.model)

  tensorboard_dir = join(FLAGS.model_dir, "memory")
  if not exists(tensorboard_dir):
      os.mkdir(tensorboard_dir)
  jax.profiler.start_trace(tensorboard_dir)

  rng = random.PRNGKey(random_seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng, init_rng = random.split(rng)

  model, state = get_model(init_rng, input_shape, model_type, model_kwargs)

  optimizer = create_optimizer(model, learning_rate, config.weight_decay)
  del model  # Don't keep a copy of the initial model.

  # Replicate optimizer and state
  optimizer = jax_utils.replicate(optimizer)
  state = jax_utils.replicate(state)

  p_eval_step = jax.pmap(
      functools.partial(
          eval_step, num_classes=num_classes, flatten_input=flatten_input),
      axis_name='batch',
  )

  def run_eval(optimizer, state, p_eval_step, test_ds):
      test_iter = iter(test_ds)
      for _, test_batch in zip(itertools.repeat(1), test_iter):
          # pylint: disable=protected-access
          test_batch = common_utils.shard(
              jax.tree_map(lambda x: x._numpy(), test_batch))
          # pylint: enable=protected-access
          out = p_eval_step(optimizer.target, state, test_batch)
      return out

  out = run_eval(optimizer, state=state, p_eval_step=p_eval_step, test_ds=test_ds)
  out.block_until_ready()
  jax.profiler.stop_trace()

if __name__ == '__main__':
  app.run(main)
