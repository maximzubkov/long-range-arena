# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention core modules for Flax."""

from collections.abc import Iterable  # pylint: disable=g-importing-member

import warnings


from flax import struct
from flax import jax_utils
from flax.nn import base
from flax.nn import initializers
from flax.nn import stochastic
from flax.nn.linear import default_kernel_init
from flax.nn.linear import DenseGeneral
from flax.nn.attention import dot_product_attention
from flax.nn.attention import Cache
from flax.nn.attention import _CacheEntry
from positional_bias.jax import name2model
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as onp

from lra_benchmarks.models.layers.common_layers import Embed


class MultiHeadDotProductAttention(base.Module):
  """Multi-head dot-product attention."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            attention_axis=None,
            causal_mask=False,
            padding_mask=None,
            key_padding_mask=None,
            segmentation=None,
            key_segmentation=None,
            cache=None,
            broadcast_dropout=True,
            dropout_rng=None,
            dropout_rate=0.,
            deterministic=False,
            precision=None,
            kernel_init=default_kernel_init,
            bias_init=initializers.zeros,
            bias=True,
            attention_fn=dot_product_attention,
            qk_transform_fn=None,
            pos_bias_cfg=None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
      inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]`
        or None for self-attention, inn which case key/values will be derived
        from inputs_q.
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      attention_axis: axes over which the attention is applied ( 'None' means
        attention over all axes, but batch, heads, and features).
      causal_mask: boolean specifying whether to apply a causal mask on the
        attention weights. If True, the output at timestep `t` will not depend
        on inputs at timesteps strictly greater than `t`.
      padding_mask: boolean specifying query tokens that are pad token.
      key_padding_mask: boolean specifying key-value tokens that are pad token.
      segmentation: segment indices for packed inputs_q data.
      key_segmentation: segment indices for packed inputs_kv data.
      cache: an instance of `flax.nn.attention.Cache` used for efficient
        autoregressive decoding.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      qk_transform_fn: A function used to transform queries and keys
        after the projection.
      pos_bias_cfg: dict or None, config for positional bias

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    assert causal_mask or not cache, (
        'Caching is only support for causal attention.')

    if inputs_kv is None:
      inputs_kv = inputs_q

    if attention_axis is None:
      attention_axis = tuple(range(1, inputs_q.ndim - 1))

    features = out_features or inputs_q.shape[-1]
    qkv_features = qkv_features or inputs_q.shape[-1]

    assert qkv_features % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // num_heads

    dense = DenseGeneral.partial(
        axis=-1,
        features=(num_heads, head_dim),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        precision=precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]
    query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                         dense(inputs_kv, dtype=dtype, name='key'),
                         dense(inputs_kv, dtype=dtype, name='value'))

    # Apply key/query transform
    if qk_transform_fn is not None:
      query, key = qk_transform_fn(query, key)

    if cache:
      assert isinstance(cache, Cache), 'cache must be an instance of Cache'
      if self.is_initializing():
        cache.store(onp.array((key.ndim,) + key.shape[-2:], dtype=onp.int32))
      else:
        cache_entry = cache.retrieve(None)
        expected_shape = list(cache_entry.key.shape[:-2])
        for attn_dim in attention_axis:
          expected_shape[attn_dim] = 1
        expected_shape = tuple(expected_shape) + inputs_q.shape[-1:]
        if expected_shape != inputs_q.shape:
          raise ValueError('Invalid shape provided, '
                           'expected shape %s instead got %s.' %
                           (expected_shape, inputs_q.shape))

        if not isinstance(cache_entry, _CacheEntry):
          raise ValueError('Cache is not initialized.')

        cshape = cache_entry.key.shape
        indices = [0] * len(cshape)
        i = cache_entry.i
        attn_size = onp.prod(onp.take(cshape, attention_axis))
        for attn_dim in attention_axis:
          attn_size //= cshape[attn_dim]
          indices[attn_dim] = i // attn_size
          i = i % attn_size

        key = lax.dynamic_update_slice(cache_entry.key, key, indices)
        value = lax.dynamic_update_slice(cache_entry.value, value, indices)
        one = jnp.array(1, jnp.uint32)
        cache_entry = cache_entry.replace(i=cache_entry.i + one,
                                          key=key,
                                          value=value)
        cache.store(cache_entry)

        # TODO(levskaya): verify this is still needed in translation decoding.
        key_padding_mask = jnp.broadcast_to(
            (jnp.arange(cshape[1]) < cache_entry.i), cshape[:2])
        key_padding_mask = key_padding_mask.astype(jnp.float32)[..., None]

    # create attention masks
    mask_components = []

    if causal_mask:
      if cache and not self.is_initializing():
        bias_pre_shape = (1,) * (key.ndim - 1)
        attn_shape = tuple(onp.take(key.shape, attention_axis))
        attn_size = onp.prod(attn_shape)
        ii = jnp.arange(attn_size, dtype=jnp.uint32)
        mask = ii < cache_entry.i
        mask_components.append(mask.reshape(bias_pre_shape + attn_shape))
      else:
        mask_components.append(_make_causal_mask(key, attention_axis))

    if padding_mask is not None:
      if key_padding_mask is None:
        key_padding_mask = padding_mask
      padding_mask = make_padding_mask(
          padding_mask_query=padding_mask,
          padding_mask_key=key_padding_mask,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis)
      mask_components.append(padding_mask)

    if segmentation is not None:
      if key_segmentation is None:
        key_segmentation = segmentation
      segmentation_mask = make_padding_mask(
          padding_mask_query=segmentation,
          padding_mask_key=key_segmentation,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis,
          segmentation_mask=True)
      mask_components.append(segmentation_mask)

    if mask_components:
      attention_mask = mask_components[0]
      for component in mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)

      # attention mask in the form of attention bias
      attention_bias = lax.select(
          attention_mask > 0, jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    if pos_bias_cfg is not None:
      if pos_bias_cfg["pos_bias_type"] == "relative_key_query":
        seq_length = pos_bias_cfg["max_seq_len"]
        position_ids_l = jnp.reshape(jnp.arange(seq_length + 1), (-1, 1))
        position_ids_r = jnp.reshape(jnp.arange(seq_length + 1), (1, -1))
        distance = position_ids_l - position_ids_r
        positional_embedding = Embed(
          distance + seq_length,
          num_embeddings=2 * seq_length + 1,
          features=head_dim
        )
        relative_position_scores_query = jnp.einsum("blhd,lrd->bhlr", query, positional_embedding)
        relative_position_scores_key = jnp.einsum("brhd,lrd->bhlr", key, positional_embedding)
        rp = relative_position_scores_query + relative_position_scores_key
        attention_bias = attention_bias + rp if attention_bias is not None else rp

    # apply attention
    x = attention_fn(
        query,
        key,
        value,
        dtype=dtype,
        axis=attention_axis,
        bias=attention_bias,
        precision=precision,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic)

    if pos_bias_cfg is not None:
      if pos_bias_cfg["pos_bias_type"] in ["fft_2d", "fft"]:
        pbv = name2model[pos_bias_cfg["pos_bias_type"]](value, **pos_bias_cfg)
        x = pbv + x

    # back to the original inputs dimensions
    out = DenseGeneral(
        x,
        features=features,
        axis=(-2, -1),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        dtype=dtype,
        precision=precision,
        name='out')

    return out


# TODO(flax-dev): Consider refactoring MultiHeadDotProductAttention and moving
# causal_mask and cache support into this class instead.
SelfAttention = MultiHeadDotProductAttention.partial(inputs_kv=None)


def make_padding_mask(padding_mask_query,
                      padding_mask_key,
                      query_shape,
                      key_shape,
                      attention_axis=None,
                      segmentation_mask=False):
  """Makes padding mask for attention weights.

  In case of 1d inputs (i.e., `[bs, len, features]`, the attention weights will
  be `[bs, len, len]` and this function makes a square matrix [len, len].

  Args:
    padding_mask_query: padding mask of query <bs, qdim1,.., qdimn>
    padding_mask_key: padding mask of query <bs, key1,.., keyn>
    query_shape: shape of the query
    key_shape: shape of the key, which is equal to the shape of value.
    attention_axis: axis over which attention is applied.
    segmentation_mask: bool: if true use equality on cartesian product rather
      than outer product for constructing segmentation masks.
  Returns:
    The padding mask for attention weights.
  """
  assert query_shape[0] == key_shape[0]
  assert len(query_shape) == len(key_shape)

  ndim = len(key_shape)
  if attention_axis is None:
    attention_axis = tuple(range(1, ndim - 2))
  assert isinstance(attention_axis, tuple)
  for ax in attention_axis:
    if not (ndim >= 3 and 1 <= ax < ndim - 2):
      raise ValueError(
          'Attention axis must be between the batch axis and the last-two axes.'
      )

  mask_shape_final = (query_shape[0], 1)  #  batch_size, 1 (for all heads)s
  for ax in attention_axis:
    mask_shape_final += (query_shape[ax],)
  for ax in attention_axis:
    mask_shape_final += (key_shape[ax],)

  padding_mask_query = padding_mask_query[..., None]
  padding_mask_key = padding_mask_key[..., None]
  perm = (0,) + tuple(onp.flip(onp.arange(padding_mask_key.ndim)))[:-1]
  if segmentation_mask:
    mask = jnp.equal(padding_mask_query, padding_mask_key.transpose(perm))
  else:
    mask = jnp.multiply(padding_mask_query, padding_mask_key.transpose(perm))

  mask = mask.reshape(mask_shape_final)
  mask = jax.lax.convert_element_type(mask, jnp.float32)
  return mask


def _make_causal_mask(key, attention_axis=None, self_mask=False):
  """Makes a causal mask, to be used for masking out the future for attention.

  In case of 1d inputs (i.e., `[bs, len, features]`, the attention weights will
  be `[bs, len, len]` and this function makes a square matrix [len, len] with
  zeros in upper triangle and ones in lower triangle.

  Args:
    key: shape of the key, which is equal to the shape of value and is
      assumed to be equal to the shape of the query (since this is used in
      self-attention when decoding).
    attention_axis: axis over which attention is applied.
    self_mask: if mask out the diagonal or not.

  Returns:
    A causal mask to be used to mask out future positions.
  """
  if attention_axis is None:
    attention_axis = tuple(range(1, key.ndim - 2))
  assert isinstance(attention_axis, tuple)
  for ax in attention_axis:
    if not (key.ndim >= 3 and 1 <= ax < key.ndim - 2):
      raise ValueError(
          'Attention axis must be between the batch axis and the last-two axes.'
      )

  mask_shape = tuple([1] * (key.ndim - len(attention_axis) - 1))
  mask_shape_final = mask_shape
  for _ in range(2):
    flatten_dim = 1
    for ax in attention_axis:
      mask_shape_final += (key.shape[ax],)
      flatten_dim *= key.shape[ax]
    mask_shape += (flatten_dim,)

  def tri(n, m, k=0):
    # Tie in the key to avoid the mask becoming a constant.
    # This way XLA can construct the mask during computation and fuse it
    # with the attention ops.
    x = lax.tie_in(key, jnp.arange(n, dtype=jnp.int32))
    y = lax.tie_in(key, jnp.arange(m, dtype=jnp.int32))
    mask = lax.ge(
        (lax.broadcast_in_dim(x, shape=(n, m), broadcast_dimensions=(0,))) + k,
        lax.broadcast(y, [n]))
    return mask

  k = -1 if self_mask else 0
  mask = tri(*mask_shape[-2:], k=k).reshape(mask_shape_final)
  return mask
