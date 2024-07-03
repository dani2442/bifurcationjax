import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib as mpl

from bifurcationjax.utils.Branch import Diagram

def plot_bifurcation_diagram(diagram: Diagram, axis: int = 0, plot_fn = lambda p: p.z[0], path_save: str | None = None):
    dict_color = {'bp':0, 'hopf':1, 'nd':2}
    cmap = plt.get_cmap()
    fig, ax = plt.subplots(figsize=(8,6))

    for bp in diagram.bps.keys():
        ax.scatter(bp.z[-1], plot_fn(bp), c=cmap(dict_color[bp.tp]), label=bp.tp)
    cmap = mpl.colormaps['Blues']

    b_len = len(diagram.branches)
    for i, branch in enumerate(diagram.branches):
        xs = [plot_fn(p) for p in branch.points]
        ps = [p.z[-1] for p in branch.points]

        ax.plot(ps, xs, color=cmap((b_len - i)/b_len), label=f"Branch $n={i+1}$")
        ax.scatter(ps, xs)
    #ax.axhline(y=0, color='r', label='Trivial Solution')
    #ax.scatter([(i**2)*(3.1415)**2 for i in range(1,5)], [0]*4, label='Bifurcation Points', zorder=41)
    #ax.set_ylim((-1,1))
    plt.grid()
    plt.legend()
    
    if path_save:
        plt.savefig('images/bifurcation_diagram_global_v4.pdf')
    plt.show()