from abc import ABC, abstractmethod
from typing import Any, Optional, Callable
from functools import partial
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm

class Predictor(ABC):
    @abstractmethod
    def __call__(self, z: jax.Array, ds: float, f: Callable) -> Any:
        pass


class NaturalPredictor(Predictor):
    def __call__(self, z: jax.Array, ds: float, f: Callable) -> jax.Array:
        z[...,-1] += ds
        return z, jnp.eye(len(z), M=1, k=1-len(z))


class SecantPredictor(Predictor):
    def __init__(self, dz0: jax.Array) -> None:
        self.dz0 = dz0
        self.prev = None

    def __call__(self, z: jax.Array, ds: float, f: Callable) -> jax.Array:
        if self.prev is None:
            v = self.dz0
        else:
            v = (z-self.prev)
        v = v/norm(v, ord=2)

        z_new = z + ds*v
        self.prev = z
        return z_new, v


class TangentPredictor(Predictor):
    def __init__(self, dz0: Optional[jax.Array] = None, k: Optional[int] = 0) -> None:
        self.dz0 = dz0
        self.prev_v = None
        self.k = k

    def __call__(self, z: jax.Array, ds: float, f: Callable) -> jax.Array:
        x, p = z[:-1], z[-1]
        j = jax.jacobian(f, argnums=0)(x, p)
        pder = jax.jacobian(f, argnums=1)(x, p)
        Jmixed = jnp.column_stack([j, pder])
        if self.prev_v is None:
            last_row = jnp.eye(len(z), M=1, k=-self.k)
            Jfinal = jnp.concat([Jmixed, last_row.T])
            v = Jfinal@jnp.eye(len(z), M=1, k=1-len(z))
            if self.dz0 is not None:
                if jnp.inner(self.dz0, v.reshape(-1))<0: v = -v
        else:
            last_row = self.prev_v.reshape(-1, 1)
            Jfinal = jnp.concat([Jmixed, last_row.T])
            v = Jfinal@jnp.eye(len(z), M=1, k=1-len(z))
        
        v = v.reshape(-1)/norm(v, ord=2)
        z_new = z + v*ds
        self.prev_v = v

        return z_new, v


class BorderedPredictor(Predictor):
    pass

