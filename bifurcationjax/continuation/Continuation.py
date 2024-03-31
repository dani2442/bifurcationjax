import jax
import jax.numpy as jnp
from typing import Tuple

from bifurcationjax.BifurcationProblem import BifurcationProblem
from bifurcationjax.continuation.Corrector import Corrector, NaturalCorrector, PALC
from bifurcationjax.continuation.Predictor import Predictor, TangentPredictor
from bifurcationjax.utils.branch_switching import normal_orthogonal_direction_method
from bifurcationjax.utils.Branch import Point, Branch, Diagram, ContinuationPar
from bifurcationjax.utils import get_bifurcation_type, is_stable


def continuation(prob: BifurcationProblem, prediction: Predictor, correction: Corrector, par: ContinuationPar, max_depth: int = 2):
    p0 = Point(step=0)
    J = jax.jit(jax.jacobian(prob.f))
    z = jnp.append(prob.x0, prob.p0)

    p0.z, success = NaturalCorrector(delta=0.1, k=1)(z, None, prob.f, None, par.dsmax)
    
    p0.evals, p0.evecs = jnp.linalg.eig(J(p0.z[:-1], p0.z[-1])) 
    p0.n_unstable, p0.n_imag, p0.stable = is_stable(p0.evals)

    zpred, v = prediction(p0.z, par.dsmax, prob.f)
    z, success = correction(zpred, p0.z, prob.f, v, par.dsmax)
    
    if not success:
        pass
    
    p1 = Point(z=z)
    p1.evals, p1.evecs = jnp.linalg.eig(J(p1.z[:-1], p1.z[-1])) 

    p1.n_unstable, p1.n_imag, p1.stable = is_stable(p1.evals)

    branch = Branch([p0, p1])
    _continuation(branch, prob, prediction, correction, par, max_depth)

    return Diagram

def _continuation(branch: Branch, prob: BifurcationProblem, prediction: Predictor, correction: Corrector, par: ContinuationPar, max_depth: int, depth: int = 0):
    p_initial, p1 = branch.points[-2:]
    J = jax.jit(jax.jacobian(prob.f))

    dz = p1.z - p_initial.z
    dz /= jnp.linalg.norm(dz, ord=2)

    predictor = TangentPredictor(dz0 = dz, k=1)
    corrector = PALC(delta=0.1)
    
    p0 = p1
    for i in range(par.max_steps):
        zpred, v = predictor(p0.z, par.dsmax, prob.f)
        z, success = corrector(zpred, p0.z, prob.f, v, par.dsmax)
        
        if par.p_min>zpred[-1] or par.p_max<zpred[-1]:
            break
        
        if not success:
            pass
        
        p1 = Point(step=i, z=z)

        p1.evals, p1.evecs = jnp.linalg.eig(J(p1.z[:-1], p1.z[-1])) 
        p1.n_unstable, p1.n_imag, p1.stable = is_stable(p1.evals)
        known, p1.tp = get_bifurcation_type(p0, p1)

        if p1.tp == 'bp' or depth<max_depth:
            v_b = normal_orthogonal_direction_method(prob.f, p0.z, p1.z)
            z1_b = p1.z + par.dsmax*v_b
            corrector_b = PALC(delta=0.1)
            z2_b, success_b = corrector_b(z1_b, p1.z, prob.f, v_b, par.dsmax)
            if success_b:
                p0_b = p1

                p1_b = Point(z=z2_b)
                p1_b.evals, p1_b.evecs = jnp.linalg.eig(J(p1_b.z[:-1], p1_b.z[-1])) 
                p1_b.n_unstable, p1_b.n_imag, p1_b.stable = is_stable(p1_b.evals)

                branch_new = Branch([p0_b, p1_b])
                _continuation(branch_new, prob, prediction, correction, par, max_depth, depth=depth+1)
        
        branch.add(p1)
        p0 = p1
        
    Diagram.branches.append(branch)