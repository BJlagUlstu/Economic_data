from matplotlib import pyplot as plt


def show_plot(model, labels, corr):
    plt.xlabel(labels['x'], fontsize=14, fontweight="bold")
    plt.ylabel(labels['y'], fontsize=14, fontweight="bold")

    plt.scatter(corr.values[:, 0], corr.values[:, 1], c=model.labels_, s=5)
    plt.show()
