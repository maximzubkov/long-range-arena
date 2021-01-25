import jax
import jax.numpy as jnp
from jax import lax
import jax_spe as spe


def make_spe_transform_fn(spe_cls, spe_kwargs, gated=True, shared=False):
    """Returns a function that applies the given SPE to keys and queries."""

    if shared:
        spe_encoder = spe_cls.shared(**spe_kwargs)
    else:
        spe_encoder = spe_cls.partial(**spe_kwargs)

    def transform_fn(queries, keys):
        rng_seed = lax.convert_element_type(
          jnp.ceil(jnp.sum(queries) * 10000000.0), jnp.int32)
        rng_key = jax.random.PRNGKey(rng_seed)

        qbar, kbar = spe_encoder(rng_key, keys.shape)
        if gated:
            qbar, kbar = spe.SPEGate(rng_key, (qbar, kbar))
        return spe.apply_spe(queries, qbar), spe.apply_spe(keys, kbar)

    return transform_fn
