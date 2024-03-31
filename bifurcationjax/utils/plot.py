import matplotlib.pyplot as plt
from bifurcationjax.utils.Branch import Branches

def plot_bifurcation_diagram(branches: Branches, axis: int = 0):
    dict_color = {'bp':0, 'hopf':1, 'nd':2}
    cmap = plt.get_cmap()
    fig, ax = plt.subplots()
    for i, branch in enumerate(branches.branches):
        for p in branch.points:
            if p.tp is not None:
                ax.scatter(p.z[-1], p.z[0], c=cmap(dict_color[p.tp]), label=p.tp)

        xs = [p.z[axis] for p in branch.points]
        ps = [p.z[-1] for p in branch.points]

        ax.plot(ps, xs, label=f"Branch {i}")

    plt.grid()
    plt.legend()
    plt.show()