from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, Optional
from functools import partial
import jax
import jax.numpy as jnp
from pydantic import BaseModel


class Predictor(ABC):
    @abstractmethod
    def __call__(self, z: jax.Array, ds: float, f: Callable) -> Any:
        pass

    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass


class NaturalPredictor(Predictor):
    def __call__(self, z: jax.Array, ds: float, f: Callable) -> jax.Array:
        z = z.at[...,-1].set(z[...,-1] +ds)
        return z, jnp.eye(len(z), M=1, k=1-len(z)).reshape(-1)


class SecantPredictor(Predictor):  
    def __init__(self, dz0: jax.Array, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dz0 = dz0
        self.theta = 0.5
        self.prev = None

    @partial(jax.jit,  static_argnums=(0))
    def norm(self, z):
        x, p = z[...,:-1], z[...,-1]
        return jnp.sqrt(self.theta*jnp.dot(x,x)/(jnp.linalg.norm(x, ord=2)+1e-6) + (1-self.theta)*p**2)

    def __call__(self, z: jax.Array, ds: float, f: Callable) -> jax.Array:
        if self.prev is None:
            v = self.dz0
        else:
            v = (z-self.prev)

        v = v/jnp.linalg.norm(v, ord=2)

        z_new = z + ds*v
        self.prev = z
        return z_new, v
    
    def reset(self):
        self.prev = None


class TangentPredictor(Predictor):
    def __init__(self, dz0: Optional[jax.Array] = None, k: Optional[int] = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dz0 = dz0
        self.prev_v = None
        self.theta = 0.5
        self.k = k


    @partial(jax.jit,  static_argnums=(0))
    def norm(self, z):
        x, p = z[...,:-1], z[...,-1]
        return self.theta*jnp.dot(x,x)/(jnp.linalg.norm(x, ord=2) + 1e-5) + (1-self.theta)*(p**2)

    def __call__(self, z: jax.Array, ds: float, f: Callable) -> jax.Array:
        x, p = z[:-1], z[-1]
        j = jax.jacobian(f, argnums=0)(x, p)
        pder = jax.jacobian(f, argnums=1)(x, p)
        Jmixed = jnp.column_stack([j, pder])
        if self.prev_v is None:
            last_row = jnp.eye(len(z), M=1, k=-self.k)
            Jfinal = jnp.concat([Jmixed, last_row.T])
            v = jnp.linalg.inv(Jfinal)@jnp.eye(len(z), M=1, k=1-len(z))
            if self.dz0 is not None:
                if jnp.inner(self.dz0, v.reshape(-1))<0: v = -v
        else:
            last_row = self.prev_v.reshape(-1, 1)
            Jfinal = jnp.concat([Jmixed, last_row.T])
            v = jnp.linalg.inv(Jfinal)@jnp.eye(len(z), M=1, k=1-len(z))
        
        v = v.reshape(-1)
        #v = v/self.norm(v)
        v = v.reshape(-1)/jnp.linalg.norm(v, ord=2)
        z_new = z + v*ds
        self.prev_v = v

        return z_new, v

    def reset(self):
        self.prev_v = None

class BorderedPredictor(Predictor):
    pass



class PredictorParams(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    method: str = 'tangent'
    dz0: Optional[jax.Array] = None
    prev_v: Optional[jax.Array] = None
    prev_z: Optional[jax.Array] = None
    theta: float = 0.5
    k: int = 0


    def init(self, **kwargs) -> Predictor:
        dicc = self.model_dump()
        dicc.update(kwargs)
        if dicc['method'] == 'tangent':
            return TangentPredictor(**dicc)
        elif dicc['method'] == 'secant':
            return SecantPredictor(**dicc)

