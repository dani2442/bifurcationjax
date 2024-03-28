import jax
import jax.numpy as jnp
from typing import Tuple

from bifurcationjax.BifurcationProblem import BifurcationProblem
from bifurcationjax.continuation.Corrector import Corrector, NaturalCorrector, PALC
from bifurcationjax.continuation.Predictor import Predictor, TangentPredictor
from bifurcationjax.utils.branch_switching import normal_orthogonal_direction_method

precision = 1e-4

@jax.jit
def is_stable(eigenvalues: jax.Array) -> Tuple[bool, int, int]:
    n_unstable = jnp.sum(eigenvalues.real>precision)
    n_imag = jnp.sum( (jnp.abs(eigenvalues.imag)>precision) * (eigenvalues.real > precision))
    isstable = n_unstable == 0
    return n_unstable, n_imag, isstable

def get_bifurcation_type(n_unstable, n_unstable_prev, n_imag, n_imag_prev):
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
        print("Error")
        tp = 'nd'
        known = True

    return known, tp


def continuation(prob: BifurcationProblem, prediction: Predictor, correction: Corrector, p_min: float, p_max: float, dsmax: float = 1e-2, max_steps: int = 1000):
    J = jax.jit(jax.jacobian(prob.f))
    z = jnp.append(prob.x0, prob.p0)
    dz = prob.dz0

    z, success = NaturalCorrector(delta=0.01, k=1)(z, None, prob.f, dz, dsmax)
    zs = [z]
    tps = [None]

    eigenvalues, eigenvectors = jnp.linalg.eig(J(z[:-1], z[-1])) 
    eigenvalues_list = [eigenvalues]

    n_unstable_prev, n_imag_prev, isstable = is_stable(eigenvalues)
    stability = [isstable]

    for _ in range(max_steps):
        zpred, v = prediction(z, dsmax, prob.f)
        if p_min>zpred[-1] or p_max<zpred[-1]:
            success = False
        else:
            z, success = correction(zpred, z, prob.f, v, dsmax)

        if not success: 
            print("hello")
            break
        zs += [z]

        eigenvalues = jnp.linalg.eigvals(J(z[:-1], z[-1])) 
        eigenvalues_list += [eigenvalues] 

        n_unstable, n_imag, isstable = is_stable(eigenvalues)
        known, tp = get_bifurcation_type(n_unstable, n_unstable_prev, n_imag, n_imag_prev)
        
        if tp == 'bp':
            dz_new = normal_orthogonal_direction_method(prob.f, zs[-1], zs[-2])
            z2_0 = (zs[-1] + zs[-2])/2
            z2_pred = z2_0 + dsmax*dz_new
            corrector2 = PALC(delta=0.01)
            z2, success2 = corrector2(z2_pred, z2_0, prob.f, dz_new, dsmax)
            zs += [z2]
            dz2 = z2 - z2_0
            dz2 /= jnp.linalg.norm(dz2, ord=2)
            predictor2 = TangentPredictor(dz0 = dz2)

            for i in range(1000):
                zpred2, v2 = predictor2(z2, dsmax, prob.f)
                z2, success2 = corrector2(zpred2, z2, prob.f, v2, dsmax)
                zs += [z2]

        n_unstable_prev, n_imag_prev = n_unstable, n_imag
        tps += [tp]
        stability += [isstable]

    xs = [z[:-1] for z in zs]
    ps = [z[-1] for z in zs]

    return xs, ps, stability, eigenvalues_list, tps