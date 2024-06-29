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


def _continuation_loop(branch, prob, par, forward):
    J = jax.jit(jax.jacobian(prob.f))

    if forward:
        it = iter(reversed(branch.points))
    else:
        it = iter(branch.points)
    p1 = next(it)
    p_initial = next(it)
    dz = p1.z - p_initial.z

    #dz /= jnp.linalg.norm(dz, ord=2)

    predictor = TangentPredictor(dz0 = dz, k=5)
    corrector = PALC(delta=0.1)
    
    bps = []
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

        if p1.tp == 'bp':
            is_bp = Diagram.get_bp(p1)
            if is_bp is None:
                Diagram.bps[p1] = {branch.id}
                bps.append([p0, p1])
            else:
                Diagram.bps[is_bp].update({branch.id})

        branch.add(p1, forward=forward)
        p0 = p1

    return bps


def _continuation(branch: Branch, prob: BifurcationProblem, prediction: Predictor, correction: Corrector, par: ContinuationPar, max_depth: int, depth: int = 1):    
    J = jax.jit(jax.jacobian(prob.f))

    bps = []
    bps += _continuation_loop(branch, prob, par, forward=True)
    bps += _continuation_loop(branch, prob, par, forward=False)

    Diagram.branches.append(branch)

    if depth >= max_depth: return 

    for p0,p1 in bps:
        v_b = normal_orthogonal_direction_method(prob.f, p0.z, p1.z)
        
        z1_b = p1.z + par.dsmax*v_b
        z1_c = p1.z - par.dsmax*v_b
        corrector_b = PALC(delta=0.1)
        
        z2_b, success_b = corrector_b(z1_b, p1.z, prob.f, v_b, par.dsmax)
        z2_c, success_c = corrector_b(z1_c, p1.z, prob.f, -v_b, par.dsmax)
        if success_b and success_c:
            p0_b = p1

            p1_b = Point(z=z2_b)
            p1_c = Point(z=z2_c)

            p1_b.evals, p1_b.evecs = jnp.linalg.eig(J(p1_b.z[:-1], p1_b.z[-1])) 
            p1_b.n_unstable, p1_b.n_imag, p1_b.stable = is_stable(p1_b.evals)

            p1_c.evals, p1_c.evecs = jnp.linalg.eig(J(p1_c.z[:-1], p1_c.z[-1])) 
            p1_c.n_unstable, p1_c.n_imag, p1_c.stable = is_stable(p1_c.evals)

            branch_new = Branch([p1_c, p0_b, p1_b])
            _continuation(branch_new, prob, prediction, correction, par, max_depth, depth=depth+1)
    
