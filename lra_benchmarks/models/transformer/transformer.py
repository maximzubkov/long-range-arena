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

"""Transformer model."""
from flax import nn
import jax.numpy as jnp
from lra_benchmarks.models.layers import common_layers
from lra_benchmarks.models.layers import attention


class TransformerBlock(nn.Module):
  """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            cache=None,
            attention_fn=None,
            qk_transform_fn=None,
            pos_bias_cfg=None):
    """Applies TransformerBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32).
      inputs_segmentation: input segmentation info for packed examples.
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      cache: flax autoregressive cache for fast decoding.
      qk_transform_fn: A function used to transform queries and keys.
      pos_bias_cfg: dict or None, config for positional bias

    Returns:
      output after transformer block.

    """

    if attention_fn is None:
      attention_fn = nn.attention.dot_product_attention

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs)
    x = attention.SelfAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=causal_mask,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache,
        attention_fn=attention_fn,
        qk_transform_fn=qk_transform_fn,
        pos_bias_cfg=pos_bias_cfg)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(x)
    y = common_layers.MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


class TransformerEncoder(nn.Module):
  """Transformer Model Encoder."""

  def apply(self,
            inputs,
            vocab_size,
            inputs_positions=None,
            inputs_segmentation=None,
            shared_embedding=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            dtype=jnp.float32,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,
            train=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            learn_pos_emb=False,
            classifier=False,
            classifier_pool='CLS',
            num_classes=10,
            tied_weights=False,
            attention_fn=None,
            add_pos_emb=True,
            qk_transform_fn=None,
            qk_transform_fn_factory=None,
            pos_bias_cfg=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      learn_pos_emb: boolean, if learn the positional embedding or use the
        sinusoidal positional embedding.
      classifier: boolean, for classification mode (output N-class logits)
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.
      tied_weights: bool, to tie weights or not.
      add_pos_emb: bool, whether to add positional embedding.
      qk_transform_fn: A function used to transform queries and keys.
      qk_transform_fn_factory: A function returning a function used to
        transform queries and keys. Use instead of `qk_transform_fn` if
        you need to create shared modules.
      pos_bias_cfg: dict or None, config for positional bias

    Returns:
      output of a transformer encoder or logits if classifier_mode is true.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[..., None]

    # Input Embedding
    if shared_embedding is None:
      input_embed = nn.Embed.partial(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)

    if classifier and classifier_pool == 'CLS':
      cls = self.param('cls', (1, 1, emb_dim), nn.initializers.zeros)
      cls = jnp.tile(cls, [x.shape[0], 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
      max_len += 1
      src_padding_mask = jnp.concatenate(
          [src_padding_mask[:, :1], src_padding_mask], axis=1)

    if add_pos_emb:
      pe_init = nn.initializers.normal(stddev=0.02) if learn_pos_emb else None
      x = common_layers.AddPositionEmbs(
          x,
          inputs_positions=inputs_positions,
          posemb_init=pe_init,
          max_len=max_len,
          name='posembed_input')
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    if qk_transform_fn_factory is not None:
      qk_transform_fn = qk_transform_fn_factory()

    if use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input Encoder
    if tied_weights:
      encoder = TransformerBlock.shared(
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          attention_fn=attention_fn,
          qk_transform_fn=qk_transform_fn,
          deterministic=not train,
          name='encoderblock',
          pos_bias_cfg=pos_bias_cfg)
      for _ in range(num_layers):
        x = encoder(x)
    else:
      for lyr in range(num_layers):
        x = TransformerBlock(
            x,
            qkv_dim=qkv_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dtype=dtype,
            padding_mask=src_padding_mask,
            inputs_segmentation=inputs_segmentation,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            attention_fn=attention_fn,
            qk_transform_fn=qk_transform_fn,
            deterministic=not train,
            name=f'encoderblock_{lyr}',
            pos_bias_cfg=pos_bias_cfg)

    encoded = nn.LayerNorm(x, dtype=dtype, name='encoder_norm')

    if classifier:
      encoded = common_layers.classifier_head(
          encoded, num_classes, mlp_dim, pooling_mode=classifier_pool)
    return encoded


class TransformerDualEncoder(nn.Module):
  """Transformer Model for Matching (dual encoding) tasks."""

  def apply(self,
            inputs1,
            inputs2,
            vocab_size=None,
            inputs1_positions=None,
            inputs2_positions=None,
            inputs1_segmentation=None,
            inputs2_segmentation=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            attention_fn=None,
            add_pos_emb=True,
            qk_transform_fn=None,
            qk_transform_fn_factory=None,
            classifier=True,
            classifier_pool='CLS',
            num_classes=2,
            interaction=None,
            pos_bias_cfg=None):
    """Applies Transformer model on text similarity.

    A deliberate choice to distinguish this from NLI because
    we may want to do different things to the model later. Dual Encoding
    mode enforces that we do not do cross attention between pairs.

    Args:
      inputs1: input data.
      inputs2: target data.
      vocab_size: size of the input vocabulary.
      inputs1_positions: input subsequence positions for packed examples.
      inputs2_positions: target subsequence positions for packed examples.
      inputs1_segmentation: input segmentation info for packed examples.
      inputs2_segmentation: target segmentation info for packed examples.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding.
      num_heads: number of heads.
      num_layers: number of layers.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      max_len: maximum length.
      train: whether it is training.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate for attention weights.
      classifier: boolean, to use classifier.
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.
      interaction: str, supports "NLI"
      add_pos_emb: bool, whether to add positional embedding.
      qk_transform_fn: A function used to transform queries and keys.
      qk_transform_fn_factory: A function returning a function used to
        transform queries and keys. Use instead of `qk_transform_fn` if
        you need to create shared modules.
      pos_bias_cfg: dict or None, config for positional bias

    Returns:
      output of a transformer decoder.
    """

    encoder = TransformerEncoder.shared(
        vocab_size=vocab_size,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        attention_fn=attention_fn,
        add_pos_emb=add_pos_emb,
        qk_transform_fn=qk_transform_fn,
        qk_transform_fn_factory=qk_transform_fn_factory,
        name='encoder',
        pos_bias_cfg=pos_bias_cfg)
    inputs1_encoded = encoder(
        inputs=inputs1,
        inputs_positions=inputs1_positions,
        inputs_segmentation=inputs1_segmentation)
    inputs2_encoded = encoder(
        inputs=inputs2,
        inputs_positions=inputs2_positions,
        inputs_segmentation=inputs2_segmentation)

    encoded = common_layers.classifier_head_dual(
        inputs1_encoded,
        inputs2_encoded,
        num_classes,
        mlp_dim,
        pooling_mode=classifier_pool,
        interaction=interaction)

    return encoded
