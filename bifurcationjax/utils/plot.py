import matplotlib.pyplot as plt
from bifurcationjax.utils.Branch import Diagram

def plot_bifurcation_diagram(diagram: Diagram, axis: int = 0):
    dict_color = {'bp':0, 'hopf':1, 'nd':2}
    cmap = plt.get_cmap()
    fig, ax = plt.subplots()

    for bp in diagram.bps.keys():
        ax.scatter(bp.z[-1], bp.z[0], c=cmap(dict_color[bp.tp]), label=bp.tp)

    for i, branch in enumerate(diagram.branches):
        xs = [p.z[axis] for p in branch.points]
        ps = [p.z[-1] for p in branch.points]

        ax.plot(ps, xs, label=f"Branch {i}")

    plt.grid()
    plt.legend()
    plt.savefig('images/bifurcation_diagram.png', dpi=300)
    plt.show()