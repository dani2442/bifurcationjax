import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from bifurcationjax.continuation.Corrector import NaturalCorrector
from bifurcationjax.continuation.Predictor import TangentPredictor
from bifurcationjax.continuation.Continuation import continuation
from bifurcationjax.BifurcationProblem import BifurcationProblem
from bifurcationjax.utils.plot import plot_bifurcation_diagram
from bifurcationjax.utils.Branch import ContinuationPar

@jax.jit
def F(u, p):
    return -u* (p + u*(2-5*u))*(p - 0.15 - u*(2+20*u))

p0 = -0.2
x0 = jnp.array([0.])

prob = BifurcationProblem(F, x0, p0,)
par = ContinuationPar(p_min=-1, p_max=1., dsmax=0.03)
correction = NaturalCorrector(k=1)
prediction = TangentPredictor(k=1)
branches = continuation(prob, prediction, correction, par)

plot_bifurcation_diagram(branches)