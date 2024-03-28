import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from bifurcationjax.continuation.Corrector import NaturalCorrector
from bifurcationjax.continuation.Predictor import TangentPredictor
from bifurcationjax.continuation.Continuation import continuation
from bifurcationjax.BifurcationProblem import BifurcationProblem

@jax.jit
def F(x, p):
    return x*(p-x)

p_min = -1
p_max = 1
p0 = -0.2
x0 = jnp.array([0.])

prob = BifurcationProblem(F, x0, p0,)
correction = NaturalCorrector(k=1)
prediction = TangentPredictor(k=1)
xs, ps, stability, eigenvalues, tps = continuation(prob, prediction, correction, p_min, p_max, dsmax=0.001)


dict_color = {'bp':0, 'hopf':1, 'nd':2}
cmap = plt.get_cmap()
fig, ax = plt.subplots()
for tp, p, x in zip(tps, ps, xs):
    if tp is not None:
        ax.scatter(p,x[0], c=cmap(dict_color[tp]), label=tp)
ax.scatter(ps, [x[0] for x in xs])
plt.grid()
plt.legend()
plt.show()