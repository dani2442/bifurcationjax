from bifurcationjax.utils.Branch import Point
import jax
import jax.numpy as jnp
from typing import Tuple

precision = 1e-4

def get_bifurcation_type(p0: Point, p1: Point):
    n_unstable, n_unstable_prev = p1.n_unstable, p0.n_unstable
    n_imag, n_imag_prev = p1.n_imag, p0.n_imag

    ind_ev = n_unstable_prev if n_unstable < n_unstable_prev else n_unstable
    tp = None

    delta_n_unstable = abs(n_unstable - n_unstable_prev)
    delta_n_imag = abs(n_imag - n_imag_prev)

    known = False
    # codim 1 bifurcation point detection based on eigenvalues distribution
    if delta_n_unstable == 1:
        # In this case, only a single eigenvalue crossed the imaginary axis
        # Either it is a Branch Point
        if delta_n_imag == 0:
            tp = 'bp'
        # Hopf bifurcation
        elif delta_n_imag == 1:
            tp = 'hopf'
        else:
            tp = 'nd' # Not defined bifurcation
        known = True
    elif delta_n_unstable == 2:
        if delta_n_imag == 2:
            tp = 'hopf'
        else:
            tp = 'nd'
        known = True
    elif delta_n_unstable > 2:
        tp = 'nd'
        known = True
    
    if delta_n_unstable < delta_n_imag:
        print("Unknown Bifurcation Point")
        tp = 'nd'
        known = True

    return known, tp


@jax.jit
def is_stable(eigenvalues: jax.Array) -> Tuple[bool, int, int]:
    n_unstable = jnp.sum(eigenvalues.real>precision)
    n_imag = jnp.sum( (jnp.abs(eigenvalues.imag)>precision) * (eigenvalues.real > precision))
    isstable = n_unstable == 0
    return n_unstable, n_imag, isstable