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

"""Main training script for the listops task."""
import functools
import itertools
import json
import os
import pprint
import time
from os.path import exists, join

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
from lra_benchmarks.listops import input_pipeline
from lra_benchmarks.models.transformer import transformer
from lra_benchmarks.utils import train_utils
from ml_collections import config_flags
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_string(
    'task_name',
    default='basic',
    help='Name of the task used for load training/test data.')
flags.DEFINE_string(
    'data_dir', default=None, help='Directory containing datasets.')
flags.DEFINE_bool(
    'test_only', default=False, help='Run the evaluation on the test data.')
flags.DEFINE_string(
    'results', default=None,
    help='Name of the JSON file to store the results.')


def create_model(key, flax_module, input_shape, model_kwargs):
  """Creates and initializes the model."""

  @functools.partial(jax.jit, backend='cpu')
  def _create_model(key):
    module = flax_module.partial(**model_kwargs)
    with nn.stochastic(key):
      _, initial_params = module.init_by_shape(key,
                                               [(input_shape, jnp.float32)])
      model = nn.Model(module, initial_params)
    return model

  return _create_model(key)


def create_optimizer(model, learning_rate):
  optimizer_def = optim.Adam(
      learning_rate,
      beta1=0.9,
      beta2=0.98,
      eps=1e-9,
      weight_decay=FLAGS.config.weight_decay)
  optimizer = optimizer_def.create(model)
  return optimizer


def train_step(optimizer, batch, learning_rate_fn, dropout_rng=None):
  """Perform a single training step."""
  train_keys = ['inputs', 'targets']
  (inputs, targets) = [batch.get(k, None) for k in train_keys]

  # We handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the
  # latter can add some stalls to the devices.
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(model):
    """Loss function used for training."""
    with nn.stochastic(dropout_rng):
      logits = model(inputs, train=True)
    loss, weight_sum = train_utils.compute_weighted_cross_entropy(
      logits, targets, num_classes=10, weights=None)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  return new_optimizer, new_dropout_rng


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  config = FLAGS.config
  batch_size = config.batch_size
  learning_rate = config.learning_rate
  num_train_steps = config.num_train_steps

  random_seed = config.random_seed
  model_type = config.model_type
  model_kwargs = (
      config.model_kwargs.to_dict() if 'model_kwargs' in config else {})

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  train_ds, _, _, encoder = input_pipeline.get_datasets(
      n_devices=jax.local_device_count(),
      task_name=FLAGS.task_name,
      data_dir=FLAGS.data_dir,
      batch_size=batch_size,
      max_length=config.max_length)

  vocab_size = encoder.vocab_size
  train_ds = train_ds.repeat()
  train_iter = iter(train_ds)
  max_length = config.max_length
  input_shape = (batch_size, max_length)

  model_kwargs.update({
      'vocab_size': vocab_size,
      'emb_dim': config.emb_dim,
      'num_heads': config.num_heads,
      'num_layers': config.num_layers,
      'qkv_dim': config.qkv_dim,
      'mlp_dim': config.mlp_dim,
      'max_len': config.max_length,
      'classifier': True,
      'num_classes': 10
  })

  if hasattr(config, 'attention_fn'):
      model_kwargs['attention_fn'] = config.attention_fn

  tensorboard_dir = join(FLAGS.model_dir, "memory")
  if not exists(tensorboard_dir):
    os.mkdir(tensorboard_dir)
  jax.profiler.start_trace(tensorboard_dir)

  rng = random.PRNGKey(random_seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng, init_rng = random.split(rng)
  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  dropout_rngs = random.split(rng, jax.local_device_count())

  if model_type == 'transformer':
    model = create_model(init_rng, transformer.TransformerEncoder, input_shape,
                         model_kwargs)
  else:
    raise ValueError('Model type not supported')

  optimizer = create_optimizer(model, learning_rate)
  del model  # Don't keep a copy of the initial model.

  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  learning_rate_fn = train_utils.create_learning_rate_scheduler(
      base_learning_rate=learning_rate)
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')

  batch = next(train_iter)
  batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access
  out = p_train_step(optimizer, batch, dropout_rng=dropout_rngs)
  out.block_until_ready()
  jax.profiler.stop_trace()


if __name__ == '__main__':
  app.run(main)
