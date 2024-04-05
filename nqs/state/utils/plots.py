import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from nqs.state.utils import plot_style


def plot_psi2(wf, r_min=-2, r_max=2, num_points=1000):
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


def plot_obd(file_name, nsamples, dim):
    """
    in two dimentions this is the rho(x, y) of one particle
    """
    assert dim == 2, "Only implemented for 2D"

    # get the .h5 file
    with h5py.File(f"data/{file_name}", "r") as f:
        positions = f["positions"][:]

        assert (
            positions.shape[0] == nsamples
        ), f"Expected {nsamples} samples, got {positions.shape[0]}"

    positions = positions.reshape(-1, dim)
    kde = gaussian_kde(positions.T)
    # https://stats.stackexchange.com/questions/231529/savitzky-golay-aka-hodrick-prescot-or-whittaker-henderson-vs-kernel#:~:text=Savitzky%2DGolay%20filters%20are%20used,a%20sample%20of%20x%20points.

    x_min, y_min = positions.min(axis=0)
    x_max, y_max = positions.max(axis=0)

    x_min_abs, x_max_abs = np.abs(x_min), np.abs(x_max)
    y_min_abs, y_max_abs = np.abs(y_min), np.abs(y_max)

    # to make the plot symmetric
    x_min, x_max = -max(x_min_abs, x_max_abs), max(x_min_abs, x_max_abs)
    y_min, y_max = -max(y_min_abs, y_max_abs), max(y_min_abs, y_max_abs)

    x, y = np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="BuPu", linewidth=0.5, color="grey", alpha=0.7)

    # Evaluate the KDE for x = 0
    y_line = np.linspace(y_min, y_max, 200)
    z_y_line = kde(np.vstack([np.zeros_like(y_line), y_line]))

    # Evaluate the KDE for y = 0
    x_line = np.linspace(x_min, x_max, 200)
    z_x_line = kde(np.vstack([x_line, np.zeros_like(x_line)]))

    # Project the KDE cuts onto the walls
    ax.plot(
        x_line,
        y_max * np.ones_like(x_line),
        z_x_line,
        color="grey",
        linestyle="-",
        linewidth=1.5,
    )  # Project onto xz plane
    ax.plot(
        x_min * np.ones_like(y_line),
        y_line,
        z_y_line,
        color="grey",
        linestyle="-",
        linewidth=1.5,
    )  # Project onto yz plane

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Approx. One-body Density")
    file_name = file_name.split(".")[0]
    plot_style.save(f"{file_name}_density_plot")
    plt.show()


def plot_tbd(file_name, nsamples, nparticles, dim):
    """
    two-body density using the radial probability density
    In two dimentions this is the rho(r1, r2) of two particles, 1 and 2
    """

    assert dim == 2, "Only implemented for 2D"

    # get the .h5 file
    with h5py.File(f"data/{file_name}", "r") as f:
        positions = f["positions"][:]

        assert (
            positions.shape[0] == nsamples
        ), f"Expected {nsamples} samples, got {positions.shape[0]}"

    positions = positions.reshape(-1, nparticles, dim)

    # Calculate the distance between the two particles for each sample
    distances = np.linalg.norm(positions[:, 0, :] - positions[:, 1, :], axis=1)

    # Use KDE to estimate the probability density of these distances
    kde = gaussian_kde(distances)

    # Define the range for the distance values
    distance_min, distance_max = distances.min(), distances.max()
    distance_range = np.linspace(distance_min, distance_max, 500)
    density = kde(distance_range)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.fill_between(distance_range, density, color="skyblue")

    # add vertical line at 1.253 and the mean of the distances
    plt.axvline(
        x=1.253, color="red", linestyle="--", label="Theo $\langle D \rangle$"  # noqa
    )  # TODO: THIS IS ONLY FOR THE 2D HARMONIC OSCILLATOR 2 PARTICLES
    plt.axvline(
        x=distances.mean(),
        color="green",
        linestyle="--",
        label="Sample $\langle D \rangle$",  # noqa
    )
    plt.plot(distance_range, density, color="steelblue")
    plt.title("Two-Body Distance Probability Density")
    plt.xlabel("Distance between two particles")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()

    # Save the plot
    file_name = file_name.split(".")[0]
    plot_style.save(f"{file_name}_two_body_density_plot.pdf")
    plt.show()
