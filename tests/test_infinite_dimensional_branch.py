import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from bifurcationjax.continuation.Corrector import CorrectorParams
from bifurcationjax.continuation.Predictor import PredictorParams
from bifurcationjax.continuation.Continuation import continuation
from bifurcationjax.BifurcationProblem import BifurcationProblem
from bifurcationjax.utils.Branch import ContinuationPar, Diagram
from bifurcationjax.utils.plot import plot_bifurcation_diagram


N = 40
h = 1/N
t = jnp.linspace(0,1,N)
a=1

"""
Solving -u'' = lambd u - au^3
"""

@jax.jit
def F(x, p):
    x = jax.lax.dynamic_update_slice(jnp.zeros((N,)), x, (1,))
    u_xx = (x[2:] + x[:-2] - 2*x[1:-1])/(h**2)
    return u_xx + p*x[1:-1] - a*jnp.power(x[1:-1],3)
    
@jax.jit
def norm(u, h):
    return jnp.sqrt(jnp.sum(jnp.square(u[1:] - u[:-1]))/h)

def plot_fn(p):
    if p.z[0]>0:
        return jnp.max(p.z[:-1])
    else:
        return jnp.min(p.z[:-1])

n=1
p0 = 0.
x0 = jnp.zeros((N-2,)) 

prob = BifurcationProblem(F, x0, p0,)
par = ContinuationPar(p_min=-5., p_max=120., dsmax=0.5, max_steps=500)
correction_params = CorrectorParams(method='PALC', epsilon=1e-3)
prediction_params = PredictorParams(method='tangent', k=N-2)
branches = continuation(prob, prediction_params, correction_params, par, max_depth=2)


plot_bifurcation_diagram(branches, plot_fn=plot_fn, path_save='images/bifurcation_diagram2.pdf')