import matplotlib.pyplot as plt
import numpy as np


def plot_psi2(wf, r_min=-10, r_max=10, num_points=1000):
    positions = np.linspace(r_min, r_max, num_points)

    if wf._N * wf._dim == 1:
        # random positions of the shape of positions
        pdf_values = np.zeros(num_points)

        for i, x1 in enumerate(positions):
            pdf_values[i] = wf.pdf(np.expand_dims(x1, axis=0))

        plt.plot(positions, pdf_values)
        plt.xlabel("x")
        plt.ylabel("|psi(x)|^2")

    elif wf._N * wf._dim == 2:
        pdf_values = np.zeros((num_points, num_points))

        for i, x1 in enumerate(positions):
            for j, x2 in enumerate(positions):
                r_pair = np.array([x1, x2])
                pdf_values[i, j] = wf.pdf(r_pair)

        X, Y = np.meshgrid(positions, positions)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Use a surface plot to visualize the probability density
        surf = ax.plot_surface(X, Y, pdf_values, cmap="viridis", edgecolor="none")

        # add the conditional probability as a shadow on the walls of x and y
        ax.contour(X, Y, pdf_values, zdir="x", offset=r_min, cmap="viridis")
        ax.contour(X, Y, pdf_values, zdir="y", offset=r_max, cmap="viridis")

        # Labels and title
        ax.set_xlabel("Position of Particle 1")
        ax.set_ylabel("Position of Particle 2")
        ax.set_zlabel("Probability Density")
        ax.set_title("3D Joint Probability Density Function")

        # Add a color bar which maps values to colors
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    else:
        raise NotImplementedError(
            "Plotting only implemented for 1D with 2 particles and 2D with 1 particle."
        )

    plt.show()
