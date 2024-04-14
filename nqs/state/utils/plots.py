import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from nqs.state.utils import plot_style
from scipy.ndimage import gaussian_filter


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

def plot_obd(file_name, nsamples, dim, method="gaussian"):
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
    x_min, y_min = positions.min(axis=0)
    x_max, y_max = positions.max(axis=0)

    x_min_abs, x_max_abs = np.abs(x_min), np.abs(x_max)
    y_min_abs, y_max_abs = np.abs(y_min), np.abs(y_max)

    # to make the plot symmetric
    x_min, x_max = -max(x_min_abs, x_max_abs), max(x_min_abs, x_max_abs)
    y_min, y_max = -max(y_min_abs, y_max_abs), max(y_min_abs, y_max_abs)

    if method == "kde":
        kde = gaussian_kde(positions.T)
        # https://stats.stackexchange.com/questions/231529/savitzky-golay-aka-hodrick-prescot-or-whittaker-henderson-vs-kernel#:~:text=Savitzky%2DGolay%20filters%20are%20used,a%20sample%20of%20x%20points.

        x, y = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="BuPu", linewidth=0.5, color="grey", alpha=0.7)

        # Evaluate the KDE for x = 0
        y_line = np.linspace(y_min, y_max, 100)
        z_y_line = kde(np.vstack([np.zeros_like(y_line), y_line]))

        # Evaluate the KDE for y = 0
        x_line = np.linspace(x_min, x_max, 100)
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

    elif method == "gaussian":
        bins = 500 # Increase number of bins for finer resolution
        x_edges = np.linspace(x_min, x_max, bins + 1)
        y_edges = np.linspace(y_min, y_max, bins + 1)

        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(positions[:, 0], positions[:, 1], bins=(x_edges, y_edges))

        # Apply Gaussian smoothing
        sigma = [5, 5]  # Increase sigma for a smoother result
        Z = gaussian_filter(hist, sigma)

        # Prepare meshgrid for plotting
        X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
        X, Y = X.T, Y.T  # Transpose to align with histogram array

        # Plot
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, Z, cmap="BuPu", linewidth=0.5, color="grey", alpha=0.7)

        # Evaluate and plot projections onto the walls
        y_line = y_edges[:-1]
        z_y_line = Z[Z.shape[0] // 2, :]
        x_line = x_edges[:-1]
        z_x_line = Z[:, Z.shape[1] // 2]
        ax.plot(x_line, y_max * np.ones_like(x_line), z_x_line, color="grey", linestyle="-", linewidth=1.5)
        ax.plot(x_min * np.ones_like(y_line), y_line, z_y_line, color="grey", linestyle="-", linewidth=1.5)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Approx. One-body Density')

    file_name = file_name.split(".")[0]
    plot_style.save(f"{file_name}_density_plot_{method}")
    plt.show()

# WIP
# def plot_pair_correlation(file_name, nsamples, dr, max_range, dim=2):
#     """
#     Calculate the pair correlation function g(r) for a set of 2D positions.
    
#     :param file_name: The name of the .h5 file containing the positions.
#     :param nsamples: The number of samples in the file.
#     :param dr: The width of the bins for the pair correlation function.
#     :param max_range: The maximum range r to compute g(r).
#     :return: Tuple of (r_values, g_r_values)
#     """
#     # Determine the number of particles

#     assert dim == 2, "Only implemented for 2D"

#     # get the .h5 file
#     with h5py.File(f"data/{file_name}", "r") as f:
#         positions = f["positions"][:]

#         assert (
#             positions.shape[0] == nsamples
#         ), f"Expected {nsamples} samples, got {positions.shape[0]}"

#     positions = positions.reshape(-1, dim)
#     batch_size = 1000

#     r_edges = np.arange(0, max_range + dr, dr)
#     r_values = r_edges[:-1] + dr / 2
#     g_r = np.zeros_like(r_values)
    
#     num_batches = int(np.ceil(len(positions) / batch_size))

#     for i in range(num_batches):
#         start1 = i * batch_size
#         end1 = start1 + batch_size
#         for j in range(i, num_batches):  # Use j = i to avoid double counting
#             start2 = j * batch_size
#             end2 = start2 + batch_size
            
#             batch1 = positions[start1:end1]
#             batch2 = positions[start2:end2]
            
#             if i == j:
#                 # Compute within the same batch, exclude self-pairing
#                 dist_array = cdist(batch1, batch2)
#                 np.fill_diagonal(dist_array, np.inf)
#             else:
#                 # Compute between batches
#                 dist_array = cdist(batch1, batch2)

#             # Histogram the distances and update the overall g_r
#             g_r_batch, _ = np.histogram(dist_array, bins=r_edges)
#             g_r += g_r_batch

#     # Normalize the pair correlation function
#     # Calculate the density of the system for normalization
#     volume = np.max(positions[:, 0]) - np.min(positions[:, 0]) * np.max(positions[:, 1]) - np.min(positions[:, 1])
#     density = len(positions) / volume

#     # Calculate the normalization factor for a shell at distance r
#     normalization = (4 * np.pi * density * r_values * dr) * len(positions)

#     # Normalize g(r) and account for batching
#     g_r /= normalization
#     g_r /= num_batches**2


#     plt.savefig(f"{file_name}_pair_correlation.png")
#     plt.show()

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
