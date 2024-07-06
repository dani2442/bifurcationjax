import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from functools import partial

def NewtonMethod(f, x0, delta=0.1, max_iters=200, precision=1e-5):
    J = jax.jit(jax.jacobian(f))

    @jax.jit
    def _NewtonStep(x0):
        return x0 - delta*jnp.linalg.inv(J(x0)) @ f(x0)

    for _ in range(max_iters):
        x = _NewtonStep(x0)
        if jnp.sum(jnp.abs(x-x0))<precision:
            return x
        x0 = x
    print("Newton not converged")
    return x