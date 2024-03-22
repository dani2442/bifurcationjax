import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from functools import partial

@partial(jax.jit, static_argnums=(0,1))
def _NewtonStep(f, J, x0, delta):
    return x0 - delta*jnp.dot(jnp.linalg.inv(J(x0)), f(x0))

def NewtonMethod(f, x0, delta=0.1, max_iters=100, precision=1e-4):
    J = jax.jit(jax.jacobian(f))
    for _ in range(max_iters):
        x = _NewtonStep(f, J, x0, delta)
        if jnp.sum(jnp.abs(x-x0))<precision:
            return x
        x0 = x
    return x