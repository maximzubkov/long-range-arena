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


def eval_step(model, batch):
  eval_keys = ['inputs', 'targets']
  (inputs, targets) = [batch.get(k, None) for k in eval_keys]
  logits = model(inputs, train=False)
  return logits


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

  random_seed = config.random_seed
  model_type = config.model_type
  model_kwargs = (
      config.model_kwargs.to_dict() if 'model_kwargs' in config else {})

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  _, eval_ds, _, encoder  = input_pipeline.get_datasets(
      n_devices=jax.local_device_count(),
      task_name=FLAGS.task_name,
      data_dir=FLAGS.data_dir,
      batch_size=batch_size,
      max_length=config.max_length)

  vocab_size = encoder.vocab_size
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

  if model_type == 'transformer':
    model = create_model(init_rng, transformer.TransformerEncoder, input_shape, model_kwargs)
  else:
    raise ValueError('Model type not supported')

  optimizer = create_optimizer(model, learning_rate)
  del model  # Don't keep a copy of the initial model.

  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  def run_eval(eval_ds, num_eval_steps=-1):
    eval_iter = iter(eval_ds)
    if num_eval_steps == -1:
      num_iter = itertools.count()
    else:
      num_iter = range(num_eval_steps)
    for _, eval_batch in zip(num_iter, eval_iter):
      # pylint: disable=protected-access
      eval_batch = common_utils.shard(
          jax.tree_map(lambda x: x._numpy(), eval_batch))
      # pylint: enable=protected-access
      out = p_eval_step(optimizer.target, eval_batch)
    return out

  out = run_eval(eval_ds, 1000)
  out.block_until_ready()
  jax.profiler.stop_trace()


if __name__ == '__main__':
  app.run(main)
