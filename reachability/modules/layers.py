from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResidualBlock(nnx.Module):
    def __init__(self, width: int, dropout: float, act: Callable, rngs: nnx.Rngs) -> None:
        self.norm = nnx.LayerNorm(width, rngs=rngs)
        self.linear1 = nnx.Linear(width, width, rngs=rngs)
        self.linear2 = nnx.Linear(width, width, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None
        self.act = act

        params_key = rngs.params()
        
        self.linear1.kernel.value = jax.nn.initializers.orthogonal()(
            params_key, self.linear1.kernel.shape
        )
        self.linear1.bias.value = jnp.zeros(self.linear1.bias.shape)

        self.linear2.kernel.value = jnp.zeros(self.linear2.kernel.shape)
        self.linear2.bias.value = jnp.zeros(self.linear2.bias.shape)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.norm(x)
        y = self.linear1(y)
        if self.dropout:
            y = self.dropout(y)
        y = self.act(y)
        y = self.linear2(y)
        
        return x + y


class ProprioHead(nnx.Module):
    def __init__(self, in_dim: int, embed_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.net = nnx.Sequential(
            nnx.Linear(in_dim, embed_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(embed_dim, embed_dim, rngs=rngs),
            nnx.LayerNorm(embed_dim, rngs=rngs),
        )

    def __call__(self, prop: jax.Array) -> jax.Array:
        return self.net(prop)


class ResBlock1D(nnx.Module):
    """1D Residual Block (post-norm), channels-last (NWC).

    ``nnx.Conv`` / ``nnx.GroupNorm`` are channels-last, so the block operates on
    (B, W, C) tensors with GroupNorm(num_groups=8):
        out = act(gn1(conv1(x)));  out = gn2(conv2(out));  return act(out + shortcut(x))
    The shortcut carries its own GroupNorm, and the stride is applied to conv1
    (and the shortcut) so downsampling actually happens.
    """
    def __init__(self, in_channels, out_channels, stride, rngs: nnx.Rngs):
        super().__init__()
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=3, strides=stride,
                              padding=1, use_bias=False, rngs=rngs)
        self.gn1 = nnx.GroupNorm(out_channels, 8, rngs=rngs)

        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=3, strides=1,
                              padding=1, use_bias=False, rngs=rngs)
        self.gn2 = nnx.GroupNorm(out_channels, 8, rngs=rngs)

        self.act = nnx.silu

        # Skip connection to match dimensions if stride > 1 or channel count changes
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nnx.Sequential(
                nnx.Conv(in_channels, out_channels, kernel_size=1, strides=stride,
                         padding="VALID", use_bias=False, rngs=rngs),
                nnx.GroupNorm(out_channels, 8, rngs=rngs),
            )
        else:
            self.shortcut = nnx.identity

    def __call__(self, x):
        residual = self.shortcut(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + residual)


class ResBlock2D(nnx.Module):
    """2D Residual Block (post-norm), the 2D analogue of ``ResBlock1D``.

    Operates on channels-last (N, H, W, C) tensors (flax convention):
        out = act(gn1(conv1(x)));  out = gn2(conv2(out));  return act(out + shortcut(x))
    stride is applied to conv1 (and the 1x1 shortcut) so downsampling happens.
    """
    def __init__(self, in_channels, out_channels, stride, rngs: nnx.Rngs):
        super().__init__()
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), strides=stride,
                              padding=1, use_bias=False, rngs=rngs)
        self.gn1 = nnx.GroupNorm(out_channels, 8, rngs=rngs)

        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), strides=1,
                              padding=1, use_bias=False, rngs=rngs)
        self.gn2 = nnx.GroupNorm(out_channels, 8, rngs=rngs)

        self.act = nnx.silu

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nnx.Sequential(
                nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), strides=stride,
                         padding="VALID", use_bias=False, rngs=rngs),
                nnx.GroupNorm(out_channels, 8, rngs=rngs),
            )
        else:
            self.shortcut = nnx.identity

    def __call__(self, x):
        residual = self.shortcut(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + residual)


class TimeEmbedding(nnx.Module):
    def __init__(self, num_octaves: int = 4, T_max: float = 3.0):
        super().__init__()
        self.num_octaves = num_octaves
        self.T_max = float(T_max)
        
        self.freqs = jnp.pi * jnp.power(2.0, jnp.arange(num_octaves, dtype=jnp.float32))

    @property
    def out_dim(self) -> int:
        """Helper property to get the exact output dimension for MLP sizing."""
        return 1 + (2 * self.num_octaves)

    def __call__(self, t: jax.Array) -> jax.Array:
        """
        Args:
            t (jax.Array): Time tensor of shape (B,) or (B, 1).

        Returns:
            jax.Array: Embedded time features of shape (B, 1 + 2 * num_octaves).
        """
        if t.ndim == 1:
            t = t[..., jnp.newaxis]

        t_norm = t / self.T_max
        angles = t_norm * self.freqs

        embeddings = jnp.concatenate([
            t_norm,
            jnp.sin(angles),
            jnp.cos(angles)
        ], axis=-1)

        return embeddings