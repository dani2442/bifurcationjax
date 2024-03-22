from abc import ABC, abstractmethod
from typing import Any
import jax
import jax.numpy as jnp
from functools import partial


class Corrector(ABC):
    def __init__(self, delta: float = 0.9, max_steps: int = 200, epsilon: float = 1e-6):
        self.delta = delta
        self.max_steps = max_steps
        self.epsilon = epsilon

    @abstractmethod
    def _newton_step(self, z, zpred, f, delta, v, h):
        pass

    def __callable__(self, zpred, f, v, h):
        c = 0
        z0 = zpred
        z1 = self._newton_step(z0, zpred, f, self.delta)
        while jnp.linalg.norm(z1 - z0, ord=2)>self.epsilon:
            z0 = z1
            z1 = self._newton_step(z0, zpred, f, self.delta, v, h)
            c+=1
            if c>self.max_steps:
                print("Newton did not converge")
                return z1, False
        return z1, True


class NaturalContinuation(Corrector):
    def __init__(self, k=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    @partial(jax.jit, static_argnums=(0, 2,3))
    def _mixed_jacobian(self, z, f):
        x, p = z[:-1], z[-1]
        j = jax.jacobian(f, argnums=0)(x, p)
        pder = jax.jacobian(f, argnums=1)(x, p)
        Jmixed = jnp.column_stack([j, pder])

        last_row = jnp.eye(len(z), M=1, k=-self.k)
        Jfinal = jnp.concat([Jmixed, last_row.T])
        return Jfinal

    @partial(jax.jit, static_argnums=(2,3,4))
    def _newton_step(self, z, zpred, f, delta, v, h):
        Jfinal = self._mixed_jacobian(z, f)
        g = f(z[:-1], z[-1])
        gz = jnp.append(g, z[self.k] - zpred[self.k])
        z = z - delta*jnp.linalg.inv(Jfinal) @ gz
        return z


class PALC(Corrector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @partial(jax.jit, static_argnums=(0,2,3))
    def _mixed_jacobian(self, z, f):
        x, p = z[:-1], z[-1]
        j = jax.jacobian(f, argnums=0)(x, p)
        pder = jax.jacobian(f, argnums=1)(x, p)
        Jmixed = jnp.column_stack([j, pder])

        last_row = jnp.ones((len(z),1))
        Jfinal = jnp.concat([Jmixed, last_row.T])
        return Jfinal

    @partial(jax.jit, static_argnums=(2,3,4))
    def _newton_step(self, z, zpred, f, delta, v, h):
        Jfinal = self._mixed_jacobian(z, f)
        g = f(z[:-1], z[-1])
        gz = jnp.append(g, jnp.inner(z - zpred, v) - h)
        z = z - delta*jnp.linalg.inv(Jfinal) @ gz
        return z

class MoorePenroseContinuation(Corrector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)