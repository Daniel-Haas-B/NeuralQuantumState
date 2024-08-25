import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nqs.state.utils import plot_style

DATA_PATH = "../data/fermion_polarized"
FIG_PATH = "figs/fermion_polarized"
import pandas as pd
import glob


# def plot_psi2(wf, r_min=-2, r_max=2, num_points=1000):
#     positions = np.linspace(r_min, r_max, num_points)

#     if wf._N * wf._dim == 1:
#         # random positions of the shape of positions
#         pdf_values = np.zeros(num_points)

#         for i, x1 in enumerate(positions):
#             pdf_values[i] = wf.pdf(np.expand_dims(x1, axis=0))

#         plt.plot(positions, pdf_values)
#         plt.xlabel("x")
#         plt.ylabel("|psi(x)|^2")

#     elif wf._N * wf._dim == 2:
#         pdf_values = np.zeros((num_points, num_points))

#         for i, x1 in enumerate(positions):
#             for j, x2 in enumerate(positions):
#                 r_pair = np.array([x1, x2])
#                 pdf_values[i, j] = wf.pdf(r_pair)

#         X, Y = np.meshgrid(positions, positions)

#         # Create a 3D plot
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection="3d")

#         # Use a surface plot to visualize the probability density
#         surf = ax.plot_surface(X, Y, pdf_values, cmap="viridis", edgecolor="none")

#         # add the conditional probability as a shadow on the walls of x and y
#         ax.contour(X, Y, pdf_values, zdir="x", offset=r_min, cmap="viridis")
#         ax.contour(X, Y, pdf_values, zdir="y", offset=r_max, cmap="viridis")

#         # Labels and title
#         ax.set_xlabel("Position of Particle 1")
#         ax.set_ylabel("Position of Particle 2")
#         ax.set_zlabel("Probability Density")
#         ax.set_title("3D Joint Probability Density Function")

#         # Add a color bar which maps values to colors
#         fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#     else:
#         raise NotImplementedError(
#             "Plotting only implemented for 1D with 2 particles and 2D with 1 particle."
#         )

#     plt.show()


# def plot_2dobd(file_name, nsamples, dim):
#     """
#     2d one body density matrix
#     """
#     assert dim == 1
#     with h5py.File(f"data/{file_name}", "r") as f:
#         positions = f["positions"][:]

#         assert (
#             positions.shape[0] == nsamples
#         ), f"Expected {nsamples} samples, got {positions.shape[0]}"

#     x_min, y_min = -4.5, -4.5
#     x_max, y_max = 4.5, 4.5

#     x_min_abs, x_max_abs = np.abs(x_min), np.abs(x_max)
#     y_min_abs, y_max_abs = np.abs(y_min), np.abs(y_max)

#     # to make the plot symmetric
#     x_min, x_max = -max(x_min_abs, x_max_abs), max(x_min_abs, x_max_abs)
#     y_min, y_max = -max(y_min_abs, y_max_abs), max(y_min_abs, y_max_abs)

#     bins = 1000  # Increase number of bins for finer resolution
#     x_edges = np.linspace(x_min, x_max, bins + 1)
#     y_edges = np.linspace(y_min, y_max, bins + 1)

#     # Create 2D histogram
#     hist, x_edges, y_edges = np.histogram2d(
#         positions[:, 0], positions[:, 1], bins=(x_edges, y_edges)
#     )

#     # Apply Gaussian smoothing
#     sigma = [10, 10]  # Increase sigma for a smoother result
#     Z = gaussian_filter(hist, sigma)

#     # Prepare meshgrid for plotting
#     X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
#     X, Y = X.T, Y.T  # Transpose to align with histogram array

#     # Plot 2d one body density with colorbar, NOT A 3D PLOT
#     plt.figure()
#     plt.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="BuPu_r")
#     plt.colorbar()
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("One-body Density")
#     # add contour lines with steps of 0.1
#     plt.contour(X, Y, Z, levels=np.arange(0, Z.max(), 0.1), colors="black", alpha=0.5)

#     file_name = file_name.split(".")[0]
#     plot_style.save(f"{file_name}_obdm_plot")
#     plt.show()


# def plot_density_profile(file_name, nsamples, dim):
#     """
#     Plot the density profile of the system.
#     """
#     assert dim == 1, "Only implemented for 1D"

#     # get the .h5 file
#     with h5py.File(f"data/{file_name}", "r") as f:
#         positions = f["positions"][:]

#         assert (
#             positions.shape[0] == nsamples
#         ), f"Expected {nsamples} samples, got {positions.shape[0]}"

#     # Calculate the density profile
#     density, bins = np.histogram(positions, bins=500, density=True)
#     bin_centers = (bins[1:] + bins[:-1]) / 2
#     # use a gaussian filter to smooth the density profile
#     density = gaussian_filter(density, sigma=5)

#     # Plot the density profile
#     plt.plot(bin_centers, density, color="steelblue")
#     plt.fill_between(bin_centers, density, color=(0.5, 0.5, 0.74), alpha=0.5)
#     plt.title("Density Profile")
#     plt.xlabel("Position")
#     plt.ylabel("Density")
#     plt.grid(True)

#     # Save the plot
#     file_name = file_name.split(".")[0]
#     plot_style.save(f"{file_name}_density_profile")
#     plt.show()


# def plot_3dobd(file_name, nsamples, dim, method="gaussian"):
#     """
#     in two dimentions this is the rho(x, y) of one particle
#     """
#     assert dim == 2, "Only implemented for 2D"
#     # get the .h5 file
#     with h5py.File(f"data/{file_name}", "r") as f:
#         positions = f["positions"][:]

#         assert (
#             positions.shape[0] == nsamples
#         ), f"Expected {nsamples} samples, got {positions.shape[0]}"

#     positions = positions.reshape(-1, dim)
#     x_min, y_min = positions.min(axis=0)
#     x_max, y_max = positions.max(axis=0)

#     x_min_abs, x_max_abs = np.abs(x_min), np.abs(x_max)
#     y_min_abs, y_max_abs = np.abs(y_min), np.abs(y_max)

#     # to make the plot symmetric
#     x_min, x_max = -max(x_min_abs, x_max_abs), max(x_min_abs, x_max_abs)
#     y_min, y_max = -max(y_min_abs, y_max_abs), max(y_min_abs, y_max_abs)

#     if method == "kde":
#         kde = gaussian_kde(positions.T)
#         # https://stats.stackexchange.com/questions/231529/savitzky-golay-aka-hodrick-prescot-or-whittaker-henderson-vs-kernel#:~:text=Savitzky%2DGolay%20filters%20are%20used,a%20sample%20of%20x%20points.

#         x, y = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
#         X, Y = np.meshgrid(x, y)
#         Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection="3d")
#         ax.plot_surface(X, Y, Z, cmap="BuPu", linewidth=0.5, color="grey", alpha=0.7)

#         # Evaluate the KDE for x = 0
#         y_line = np.linspace(y_min, y_max, 100)
#         z_y_line = kde(np.vstack([np.zeros_like(y_line), y_line]))

#         # Evaluate the KDE for y = 0
#         x_line = np.linspace(x_min, x_max, 100)
#         z_x_line = kde(np.vstack([x_line, np.zeros_like(x_line)]))

#         # Project the KDE cuts onto the walls
#         ax.plot(
#             x_line,
#             y_max * np.ones_like(x_line),
#             z_x_line,
#             color="grey",
#             linestyle="-",
#             linewidth=1.5,
#         )  # Project onto xz plane
#         ax.plot(
#             x_min * np.ones_like(y_line),
#             y_line,
#             z_y_line,
#             color="grey",
#             linestyle="-",
#             linewidth=1.5,
#         )  # Project onto yz plane

#     elif method == "gaussian":
#         bins = 500  # Increase number of bins for finer resolution
#         x_edges = np.linspace(x_min, x_max, bins + 1)
#         y_edges = np.linspace(y_min, y_max, bins + 1)

#         # Create 2D histogram
#         hist, x_edges, y_edges = np.histogram2d(
#             positions[:, 0], positions[:, 1], bins=(x_edges, y_edges)
#         )

#         # Apply Gaussian smoothing
#         sigma = [5, 5]  # Increase sigma for a smoother result
#         Z = gaussian_filter(hist, sigma)

#         # Prepare meshgrid for plotting
#         X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
#         X, Y = X.T, Y.T  # Transpose to align with histogram array

#         # Plot
#         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#         ax.plot_surface(X, Y, Z, cmap="BuPu", linewidth=0.3, color="grey", alpha=0.7)

#         # Evaluate and plot projections onto the walls
#         y_line = y_edges[:-1]
#         z_y_line = Z[Z.shape[0] // 2, :]
#         x_line = x_edges[:-1]
#         z_x_line = Z[:, Z.shape[1] // 2]
#         ax.plot(
#             x_line,
#             y_max * np.ones_like(x_line),
#             z_x_line,
#             color="grey",
#             linestyle="-",
#             linewidth=1.5,
#         )
#         ax.plot(
#             x_min * np.ones_like(y_line),
#             y_line,
#             z_y_line,
#             color="grey",
#             linestyle="-",
#             linewidth=1.5,
#         )

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Approx. One-body Density")

#     file_name = file_name.split(".")[0]
#     plot_style.save(f"{file_name}_density_plot_{method}")
#     plt.show()


# def plot_tbd(file_name, nsamples, nparticles, dim):
#     """
#     two-body density using the radial probability density
#     In two dimentions this is the rho(r1, r2) of two particles, 1 and 2
#     """

#     assert dim == 2, "Only implemented for 2D"

#     # get the .h5 file
#     with h5py.File(f"data/{file_name}", "r") as f:
#         positions = f["positions"][:]

#         assert (
#             positions.shape[0] == nsamples
#         ), f"Expected {nsamples} samples, got {positions.shape[0]}"

#     positions = positions.reshape(-1, nparticles, dim)

#     # Calculate the distance between the two particles for each sample
#     distances = np.linalg.norm(positions[:, 0, :] - positions[:, 1, :], axis=1)

#     # Use KDE to estimate the probability density of these distances
#     kde = gaussian_kde(distances)

#     # Define the range for the distance values
#     distance_min, distance_max = distances.min(), distances.max()
#     distance_range = np.linspace(distance_min, distance_max, 500)
#     density = kde(distance_range)

#     # Create the plot
#     plt.figure(figsize=(8, 6))
#     plt.fill_between(distance_range, density, color="skyblue")

#     # add vertical line at 1.253 and the mean of the distances
#     plt.axvline(
#         x=1.253, color="red", linestyle="--", label="Theo $\langle D \rangle$"  # noqa
#     )  # TODO: THIS IS ONLY FOR THE 2D HARMONIC OSCILLATOR 2 PARTICLES
#     plt.axvline(
#         x=distances.mean(),
#         color="green",
#         linestyle="--",
#         label="Sample $\langle D \rangle$",  # noqa
#     )
#     plt.plot(distance_range, density, color="steelblue")
#     plt.title("Two-Body Distance Probability Density")
#     plt.xlabel("Distance between two particles")
#     plt.ylabel("Probability Density")
#     plt.grid(True)
#     plt.legend()

#     # Save the plot
#     file_name = file_name.split(".")[0]
#     plot_style.save(f"{file_name}_two_body_density_plot.pdf")
#     plt.show()


# # plot the wave function evaluated at multiple points
# def plot_psi(system, N, dim):
#     """
#     Plot the wave function evaluated at multiple points.
#     """

#     assert dim == 1, "Only implemented for 2D"
#     assert N == 2, "Only implemented for 2 particles"

#     # for 2 particles, 2 dim
#     xpoints = 100
#     r1 = np.linspace(-3, 3, xpoints)
#     r2 = np.linspace(-3, 3, xpoints)
#     X, Y = np.meshgrid(r1, r2)

#     psi = np.zeros((xpoints, xpoints))
#     for i in range(xpoints):
#         for j in range(xpoints):
#             # print("system.wf(np.array([r1[i], r2[j]]))", system.wf(np.array([r1[i], r2[j]])))

#             psi[i, j] = system.wf(np.array([r1[i], r2[j]]))

#     plt.contourf(X, Y, psi, cmap="BuPu")
#     # fix square aspect ratio
#     plt.gca().set_aspect("equal", adjustable="box")

#     plt.colorbar()
#     plot_style.save("1D2P_wave_function")
#     plt.show()


def energy_convergence_loop_side_by_side(v0_values, n_particles=2):
    """
    Plot the convergence of the energy as a function of the number of samples for multiple v0 values.
    """
    z = 1.96  # 95% confidence interval

    # Set the seaborn pastel palette
    sns.set_palette("pastel")
    palette = sns.color_palette()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    # Define color map for different nqs_types
    color_map = {}
    for v0 in v0_values:
        df = pd.read_csv(DATA_PATH + f"/energies_and_stds_loop_v0_{v0}.csv")
        nqs_types = df["nqs_type"].unique()
        color_map.update(
            {
                nqs_type: palette[i % len(palette)]
                for i, nqs_type in enumerate(nqs_types)
            }
        )

    for ax, v0 in zip(axs, v0_values):
        df = pd.read_csv(DATA_PATH + f"/energies_and_stds_loop_v0_{v0}.csv")
        nqs_types = df["nqs_type"].unique()

        for nqs_type in nqs_types:
            # Filter data for the current n_particles and nqs_type
            filtered_df = df[
                (df["n_particles"] == n_particles) & (df["nqs_type"] == nqs_type)
            ]
            if filtered_df.empty:
                continue

            # Extract energies and standard errors
            energies = filtered_df["energy"]
            std = filtered_df["std_error"]

            # get rolling mean
            energies = energies.rolling(window=10).mean()
            std = std.rolling(window=10).mean()

            stds = z * std / np.sqrt(filtered_df["batch_size"])

            # Reset index for plotting
            energies = energies.reset_index(drop=True)
            stds = stds.reset_index(drop=True)
            # make stds smoother
            stds_plus = (stds + energies).rolling(window=2).mean()
            stds_minus = (energies - stds).rolling(window=2).mean()

            # Add shaded area for the std error
            ax.fill_between(
                energies.index,
                stds_minus,
                stds_plus,
                alpha=0.2,
                color=color_map[nqs_type],
            )

            # Plot energies
            ax.plot(energies, label=f"{nqs_type}", color=color_map[nqs_type])

        # Add ground state line
        if v0 == 0:
            analytical = (n_particles**2) / 2
            ax.axhline(
                y=analytical, color="black", linestyle="--", label="Analytical", lw=1
            )
            ax.text(
                600,
                analytical,
                f"N={n_particles}",
                verticalalignment="bottom",
                horizontalalignment="right",
                fontsize=14,
            )
        else:
            try:
                df_hf = pd.read_csv(DATA_PATH + "/data_hf.csv")
                df_hf = df_hf[df_hf["A"] == n_particles]
                df_hf = df_hf[df_hf["V0"] == v0]
                e_hf = df_hf["Energy"].values[0]
                ax.axhline(y=e_hf, color="black", linestyle="-.", label="HF", lw=1)

            except Exception as e:
                raise ValueError(f"{e} No data for HF")
            try:
                df_ci = pd.read_csv(DATA_PATH + "/data_ci.csv")
                df_ci = df_ci[df_ci["A"] == n_particles]
                df_ci = df_ci[df_ci["V0"] == v0]
                e_ci = df_ci["Energy"].values[0]
                ax.axhline(y=e_ci, color="black", linestyle=":", label="CI", lw=1)
            except Exception as e:
                raise ValueError(f"{e} No data for CI")

        # Add V_0 to the top center of the plot
        ax.text(
            0.5,
            0.95,
            f"$V_0$ = {v0}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=18,
        )
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Energy")

    # Custom legend to include markers for n_particles
    handles, labels = axs[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    legend_labels = list(unique_labels.keys())
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        fontsize=18,
        bbox_to_anchor=(0.5, 1.005),
        markerscale=1.1,
        borderpad=0.6,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_style.save("fermion_polarized/energy_convergence_v0_all")


def energy_convergence_loop_side_by_side_jastrow(v0_values, n_particles=2):
    """
    Plot the convergence of the energy as a function of the number of samples for multiple v0 values.
    """
    z = 1.96  # 95% confidence interval

    # Set the seaborn pastel palette
    sns.set_palette("pastel")
    palette = sns.color_palette()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    # Define color map for different correlations
    color_map = {
        "Jastrow": palette[1],
        "No Jastrow": palette[0],
    }  # Orange for Jastrow, Blue for No Jastrow

    for ax, v0 in zip(axs, v0_values):
        df = pd.read_csv(
            DATA_PATH + f"/dsffnn_jastrow_energies_and_stds_loop_v0_{v0}.csv"
        )
        # replace None with "No Jastrow"
        df["correlation"] = df["correlation"].apply(
            lambda x: x if pd.notnull(x) else "No Jastrow"
        )
        # replace "j" with "Jastrow"
        df["correlation"] = df["correlation"].apply(
            lambda x: "Jastrow" if x == "j" else x
        )

        correlations = df["correlation"].unique()

        for correlation in correlations:
            filtered_df = df[
                (df["n_particles"] == n_particles) & (df["correlation"] == correlation)
            ]
            if filtered_df.empty:
                continue

            # Extract energies and standard errors
            energies = filtered_df["energy"]
            # get rolling mean
            energies = energies.rolling(window=10).mean()

            stds = z * filtered_df["std_error"] / np.sqrt(filtered_df["batch_size"])

            # get rolling mean
            stds = stds.rolling(window=10).mean()

            # Reset index for plotting
            energies = energies.reset_index(drop=True)
            stds = stds.reset_index(drop=True)
            # make stds smoother
            stds_plus = (stds + energies).rolling(window=2).mean()
            stds_minus = (energies - stds).rolling(window=2).mean()

            # Add shaded area for the std error
            ax.fill_between(
                energies.index,
                stds_minus,
                stds_plus,
                alpha=0.2,
                color=color_map[correlation],
            )

            # Plot energies
            ax.plot(energies, label=f"{correlation}", color=color_map[correlation])

        # Add ground state line
        if v0 == 0:
            analytical = (n_particles**2) / 2
            ax.axhline(
                y=analytical, color="black", linestyle="--", label="Analytical", lw=1
            )
            ax.text(
                600,
                analytical,
                f"N={n_particles}",
                verticalalignment="bottom",
                horizontalalignment="right",
                fontsize=14,
            )
        else:
            try:
                df_hf = pd.read_csv(DATA_PATH + "/data_hf.csv")
                df_hf = df_hf[(df_hf["A"] == n_particles) & (df_hf["V0"] == v0)]
                e_hf = df_hf["Energy"].values[0]
                ax.axhline(y=e_hf, color="black", linestyle="-.", label="HF", lw=1)
            except Exception as e:
                raise ValueError(f"{e} No data for HF")
            try:
                df_ci = pd.read_csv(DATA_PATH + "/data_ci.csv")
                df_ci = df_ci[(df_ci["A"] == n_particles) & (df_ci["V0"] == v0)]
                e_ci = df_ci["Energy"].values[0]
                ax.axhline(y=e_ci, color="black", linestyle=":", label="CI", lw=1)
            except Exception as e:
                raise ValueError(f"{e} No data for CI")

        # Add V_0 to the top center of the plot
        ax.text(
            0.5,
            0.95,
            f"$V_0$ = {v0}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=18,
        )
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Energy")

    # Custom legend to include markers for n_particles
    handles, labels = axs[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    legend_labels = list(unique_labels.keys())
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        fontsize=18,
        bbox_to_anchor=(0.5, 1.005),
        markerscale=1.1,
        borderpad=0.6,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_style.save("fermion_polarized/dsffnn_jastrow_energy_convergence_v0_all")


def energy_convergence_loop(v0, n_particles=2):
    """
    Plot the convergence of the energy as a function of the number of samples.
    """
    # Load the data
    df = pd.read_csv(DATA_PATH + f"/energies_and_stds_loop_v0_{v0}.csv")

    nqs_types = df["nqs_type"].unique()
    n_particles_values = [n_particles]  # df["n_particles"].unique()
    z = 1.96  # 95% confidence interval

    # Set the seaborn pastel palette
    sns.set_palette("pastel")
    palette = sns.color_palette()

    # Define color map for different nqs_types
    color_map = {
        nqs_type: palette[i % len(palette)] for i, nqs_type in enumerate(nqs_types)
    }

    for i, n_particles in enumerate(n_particles_values):
        for nqs_type in nqs_types:
            # Filter data for the current n_particles and nqs_type
            filtered_df = df[
                (df["n_particles"] == n_particles) & (df["nqs_type"] == nqs_type)
            ]
            if filtered_df.empty:
                continue

            # Extract energies and standard errors
            energies = filtered_df["energy"]
            stds = z * filtered_df["std_error"] / np.sqrt(filtered_df["batch_size"])

            # Reset index for plotting
            energies = energies.reset_index(drop=True)
            stds = stds.reset_index(drop=True)
            # make stds smoother
            stds_plus = (stds + energies).rolling(window=2).mean()
            stds_minus = (energies - stds).rolling(window=2).mean()

            # Add shaded area for the std error
            plt.fill_between(
                energies.index, stds_minus, stds_plus, alpha=0.1, color="purple"
            )

            # Plot energies
            plt.plot(energies, label=f"{nqs_type}", color=color_map[nqs_type])

        # Add ground state line
        if v0 == 0:
            analytical = (n_particles**2) / 2
            plt.axhline(
                y=analytical, color="black", linestyle="--", label="Analytical", lw=1
            )
            plt.text(
                600,
                analytical,
                f"N={n_particles}",
                verticalalignment="bottom",
                horizontalalignment="right",
                fontsize=14,
            )
        else:
            try:
                df_hf = pd.read_csv(DATA_PATH + "/data_hf.csv")
                df_hf = df_hf[df_hf["A"] == n_particles]
                df_hf = df_hf[df_hf["V0"] == v0]
                e_hf = df_hf["Energy"].values[0]
                plt.axhline(y=e_hf, color="black", linestyle="-.", label="HF", lw=1)

            except Exception as e:
                raise ValueError(f"{e} No data for HF")
            try:
                df_ci = pd.read_csv(DATA_PATH + "/data_ci.csv")
                df_ci = df_ci[df_ci["A"] == n_particles]
                df_ci = df_ci[df_ci["V0"] == v0]
                e_ci = df_ci["Energy"].values[0]
                plt.axhline(y=e_ci, color="black", linestyle=":", label="CI", lw=1)
            except Exception as e:
                raise ValueError(f"{e} No data for CI")

        # Add V_0 to the top center of the plot
        plt.text(
            0.5,
            0.95,
            f"$V_0$ = {v0}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            fontsize=14,
        )

    plt.xlabel("Epochs")
    plt.ylabel("Energy")

    # Custom legend to include markers for n_particles
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    legend_labels = list(unique_labels.keys())
    plt.legend(handles, legend_labels, loc="best")

    plt.grid(True)
    plt.tight_layout()
    plot_style.save(f"fermion_polarized/energy_convergence_v0_{v0}")


def plot_opti_sweep():
    def plot_graph_clean(data, title, ax):
        # get columns with MIN to be the lower bound
        # get columns with MAX to be the upper bound
        # FILL BETWEEN with shade
        # opti types
        opti_types = ["Adam", "RMSProp", "AdaGrad", "SR"]
        for opti in opti_types:
            ax.plot(data["Step"], data[f"{opti.lower()}"], label=f"{opti}")
            ax.fill_between(
                data["Step"],
                data[f"{opti.lower()}__MIN"],
                data[f"{opti.lower()}__MAX"],
                alpha=0.2,
            )

        # add fci values and hf values
        if "N = 2" in title:
            df_hf = pd.read_csv(DATA_PATH + "/data_hf.csv")
            df_hf = df_hf[df_hf["A"] == 2]
            df_hf = df_hf[df_hf["V0"] == -20]
            e_hf = df_hf["Energy"].values[0]
            ax.axhline(y=e_hf, color="black", linestyle="-.", label="HF", lw=1)
            try:
                df_ci = pd.read_csv(DATA_PATH + "/data_ci.csv")
                df_ci = df_ci[df_ci["A"] == 2]
                df_ci = df_ci[df_ci["V0"] == -20]
                e_ci = df_ci["Energy"].values[0]
                ax.axhline(y=e_ci, color="black", linestyle=":", label="CI", lw=1)
            except Exception as e:
                print(f"{e} No data for CI")

        if "N = 4" in title:
            df_hf = pd.read_csv(DATA_PATH + "/data_hf.csv")
            df_hf = df_hf[df_hf["A"] == 4]
            df_hf = df_hf[df_hf["V0"] == -20]
            e_hf = df_hf["Energy"].values[0]
            ax.axhline(y=e_hf, color="black", linestyle="-.", label="HF", lw=1)
            try:
                df_ci = pd.read_csv(DATA_PATH + "/data_ci.csv")
                df_ci = df_ci[df_ci["A"] == 4]
                df_ci = df_ci[df_ci["V0"] == -20]
                e_ci = df_ci["Energy"].values[0]
                ax.axhline(y=e_ci, color="black", linestyle=":", label="CI", lw=1)
            except Exception as e:
                print(f"{e} No data for CI")

        if "N = 6" in title:
            df_hf = pd.read_csv(DATA_PATH + "/data_hf.csv")
            df_hf = df_hf[df_hf["A"] == 6]
            df_hf = df_hf[df_hf["V0"] == -20]
            e_hf = df_hf["Energy"].values[0]
            ax.axhline(y=e_hf, color="black", linestyle="-.", label="HF", lw=1)
            try:
                df_ci = pd.read_csv(DATA_PATH + "/data_ci.csv")
                df_ci = df_ci[df_ci["A"] == 6]
                df_ci = df_ci[df_ci["V0"] == -20]
                e_ci = df_ci["Energy"].values[0]
                ax.axhline(y=e_ci, color="black", linestyle=":", label="CI", lw=1)
            except Exception as e:
                print(f"{e} No data for CI")

        ax.set_title(title)

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Energy")

    # Create subplots

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    data_n2 = pd.read_csv(DATA_PATH + "/opti_sweep_n=2.csv")
    data_n4 = pd.read_csv(DATA_PATH + "/opti_sweep_n=4.csv")
    data_n6 = pd.read_csv(DATA_PATH + "/opti_sweep_n=6.csv")
    # Plot the graphs
    plot_graph_clean(data_n2, "$V_0 = -20, | N = 2$", axs[0])  # noqa
    plot_graph_clean(data_n4, "$V_0 = -20, | N = 4$", axs[1])  # noqa
    plot_graph_clean(data_n6, "$V_0 = -20, | N = 6$", axs[2])  # noqa

    # add one legend for all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        fontsize=12,
        bbox_to_anchor=(0.5, 0.26),
        borderpad=0.5,
    )

    plt.tight_layout()
    plot_style.save("fermion_polarized/opti_sweepv0-20")


# def plot_one_body_density_matrix(n=2, nqs="dsffn"):

#         # Function to calculate the one-body density matrix using the ghost particle method
#     def calculate_one_body_density_matrix(positions, grid_points):
#         n = len(grid_points)
#         density_matrix = np.zeros((n, n))

#         # Calculate KDE for the given positions
#         kde = gaussian_kde(positions[:, 0])

#         for i, x in enumerate(grid_points):
#             for j, xp in enumerate(grid_points):
#                 # Evaluate the KDE for both x and x'
#                 rho_x = kde(x)[0]
#                 rho_xp = kde(xp)[0]
#                 # Compute the density matrix element
#                 density_matrix[i, j] = rho_x * rho_xp

#         return density_matrix

#     # Define the grid for positions
#     grid_points = np.linspace(-4, 4, 100)  # Adjust the range and resolution as needed

#     # Load the position data for different V0 values and calculate the one-body density matrices
#     V0_values = [-20, -10, 0, 10, 20]
#     file_paths = [
#         f'{DATA_PATH}/sampler/N{n}_V-20_energies_and_pos_{nqs}_ch0.h5',
#         f'{DATA_PATH}/sampler/N{n}_V-10_energies_and_pos_{nqs}_ch0.h5',
#         f'{DATA_PATH}/sampler/N{n}_V0_energies_and_pos_{nqs}_ch0.h5',
#         f'{DATA_PATH}/sampler/N{n}_V10_energies_and_pos_{nqs}_ch0.h5',
#         f'{DATA_PATH}/sampler/N{n}_V20_energies_and_pos_{nqs}_ch0.h5'
#     ]
#     plt.figure(figsize=(20, 5))

#     for i, (V0, file_path) in enumerate(zip(V0_values, file_paths)):
#         with h5py.File(file_path, 'r') as hdf:
#             positions = np.array(hdf['positions'])

#         # Calculate the one-body density matrix
#         density_matrix = calculate_one_body_density_matrix(positions, grid_points)


#         # Plot the one-body density matrix
#         plt.subplot(1, 5, i + 1)
#         plt.contourf(grid_points, grid_points, density_matrix, levels=50, cmap='Purples')
#         # make contourline around the density matrix
#         plt.contour(grid_points, grid_points, density_matrix, levels=10, colors='black', alpha=0.5)

#         plt.colorbar(label='Density')
#         plt.title(f'$V_0 = {V0}$')
#         plt.xlabel('x')
#         plt.ylabel('x\'')
#         plt.clim(0, 1)  # Adjust the color limits for consistency

#     plt.tight_layout()
#     plot_style.save("fermion_polarized/one_body_density_matrix")


def plot_density_profile(n_values, v0_values, nqs):
    # line types for different NQS types
    line_types = {"VMC": "-", "RBM": "--", "DSFFN": "-."}

    # sns pastel palette
    colors = {
        "VMC": sns.color_palette("pastel")[0],
        "RBM": sns.color_palette("pastel")[1],
        "DSFFN": sns.color_palette("pastel")[2],
    }

    def plot_position_density(positions, V0_value, n_value, nqs, ax, fontsize=14):
        # sns.kdeplot(positions[:, 0], ax=ax, color=colors[nqs], fill=False, linestyle=line_types[nqs], label=nqs)
        # elliminate crazy outliers

        hist, bin_edges = np.histogram(positions[:, 0], bins=200, density=True)
        n_psi = n_value * hist
        # force bin edges to be the same
        bin_edges = np.linspace(-4, 4, 200)
        ax.plot(
            bin_edges, n_psi, label=nqs, color=colors[nqs], linestyle=line_types[nqs]
        )

        if n_value == 3 or n_value == 2:
            ax.set_title(f"$V_0 = {V0_value}$")

        if n_value == max(n_values):
            ax.set_xlabel("Position, x")

        if V0_value == 0 and (n_value == 3 or n_value == 2):
            ax.legend()
        if V0_value == min(v0_values):
            ax.set_ylabel("$n(x)$")
            ax.text(
                0.5,
                0.9,
                f"N = {n_value}",
                horizontalalignment="right",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=fontsize,
            )

        ax.set_ylim(0, 2)
        ax.set_xlim(-4, 4)

    if nqs == "all":
        file_base_pattern_vmc = (
            f"{DATA_PATH}/sampler/N{{}}_V{{}}_energies_and_pos_VMC_ch*.h5"
        )
        file_base_pattern_rbm = (
            f"{DATA_PATH}/sampler/N{{}}_V{{}}_energies_and_pos_RBM_ch*.h5"
        )
        file_base_pattern_ds = (
            f"{DATA_PATH}/sampler/N{{}}_V{{}}_energies_and_pos_DSFFN_ch*.h5"
        )
        file_base_patterns = [
            file_base_pattern_vmc,
            file_base_pattern_rbm,
            file_base_pattern_ds,
        ]
    else:
        file_base_patterns = [
            f"{DATA_PATH}/sampler/N{{}}_V{{}}_energies_and_pos_{nqs}_ch*.h5"
        ]
    if n_values[0] == 2:
        figsize = (12, 8)
        fontsize = 16
    else:
        figsize = (10, 3)
        fontsize = 14

    fig, axes = plt.subplots(
        len(n_values), len(v0_values), figsize=figsize, squeeze=False
    )

    for file_base_pattern in file_base_patterns:
        nqs = file_base_pattern.split("_")[-2]
        for i, n in enumerate(n_values):
            print(f"Plotting density profile for N = {n}")
            for j, V0 in enumerate(v0_values):
                print(f"Plotting density profile for V0 = {V0}")
                file_pattern = file_base_pattern.format(n, V0)
                file_paths = glob.glob(file_pattern)

                if not file_paths:
                    print(f"No files found for N = {n}, V0 = {V0}")
                    continue

                all_positions = []
                print(f"Found {len(file_paths)} files for N = {n}, V0 = {V0}")
                for file_path in file_paths:
                    with h5py.File(file_path, "r") as hdf:
                        positions = np.array(hdf["positions"])
                        all_positions.append(positions)

                if all_positions:
                    combined_positions = np.vstack(all_positions)
                    plot_position_density(
                        combined_positions, V0, n, nqs, axes[i, j], fontsize=fontsize
                    )

    plt.tight_layout()
    plot_style.save(f"fermion_polarized/density_profile_N{n_values}_nqs_{nqs}")


def plot_lmh_vs_m(
    data_pth="/Users/orpheus/Documents/Masters/NeuralQuantumState/data/fermion_polarized/vmc_complete_metadata.csv",
):
    df_m_20 = pd.read_csv(DATA_PATH + "/v_20_vmc_m_all_N.csv")

    df_lmh_20 = pd.read_csv(DATA_PATH + "/v_20_vmc_lmh_all_N.csv")

    df_m_minus_20 = pd.read_csv(DATA_PATH + "/v_-20_vmc_m_all_N.csv")
    df_lmh_minus_20 = pd.read_csv(DATA_PATH + "/v_-20_vmc_lmh_all_N.csv")

    palette = sns.color_palette("pastel")

    # Extracting data for plotting
    steps_m_20 = df_m_20["Step"]
    energy_m_20_n2 = df_m_20["nparticles: 2 - energy"].rolling(window=20).mean()
    min_energy_m_20_n2 = (
        df_m_20["nparticles: 2 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_m_20_n2 = (
        df_m_20["nparticles: 2 - energy__MAX"].rolling(window=20).mean()
    )
    energy_m_20_n4 = df_m_20["nparticles: 4 - energy"].rolling(window=20).mean()
    min_energy_m_20_n4 = (
        df_m_20["nparticles: 4 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_m_20_n4 = (
        df_m_20["nparticles: 4 - energy__MAX"].rolling(window=20).mean()
    )
    energy_m_20_n6 = df_m_20["nparticles: 6 - energy"].rolling(window=20).mean()
    min_energy_m_20_n6 = (
        df_m_20["nparticles: 6 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_m_20_n6 = (
        df_m_20["nparticles: 6 - energy__MAX"].rolling(window=20).mean()
    )
    steps_lmh_20 = df_lmh_20["Step"]
    energy_lmh_20_n2 = df_lmh_20["nparticles: 2 - energy"].rolling(window=20).mean()
    min_energy_lmh_20_n2 = (
        df_lmh_20["nparticles: 2 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_lmh_20_n2 = (
        df_lmh_20["nparticles: 2 - energy__MAX"].rolling(window=20).mean()
    )
    energy_lmh_20_n4 = df_lmh_20["nparticles: 4 - energy"].rolling(window=20).mean()
    min_energy_lmh_20_n4 = (
        df_lmh_20["nparticles: 4 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_lmh_20_n4 = (
        df_lmh_20["nparticles: 4 - energy__MAX"].rolling(window=20).mean()
    )
    energy_lmh_20_n6 = df_lmh_20["nparticles: 6 - energy"].rolling(window=20).mean()
    min_energy_lmh_20_n6 = (
        df_lmh_20["nparticles: 6 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_lmh_20_n6 = (
        df_lmh_20["nparticles: 6 - energy__MAX"].rolling(window=20).mean()
    )
    steps_m_minus_20 = df_m_minus_20["Step"]
    energy_m_minus_20_n2 = (
        df_m_minus_20["nparticles: 2 - energy"].rolling(window=20).mean()
    )
    min_energy_m_minus_20_n2 = (
        df_m_minus_20["nparticles: 2 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_m_minus_20_n2 = (
        df_m_minus_20["nparticles: 2 - energy__MAX"].rolling(window=20).mean()
    )

    energy_m_minus_20_n4 = (
        df_m_minus_20["nparticles: 4 - energy"].rolling(window=20).mean()
    )
    min_energy_m_minus_20_n4 = (
        df_m_minus_20["nparticles: 4 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_m_minus_20_n4 = (
        df_m_minus_20["nparticles: 4 - energy__MAX"].rolling(window=20).mean()
    )

    energy_m_minus_20_n6 = (
        df_m_minus_20["nparticles: 6 - energy"].rolling(window=20).mean()
    )
    min_energy_m_minus_20_n6 = (
        df_m_minus_20["nparticles: 6 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_m_minus_20_n6 = (
        df_m_minus_20["nparticles: 6 - energy__MAX"].rolling(window=20).mean()
    )

    steps_lmh_minus_20 = df_lmh_minus_20["Step"]
    energy_lmh_minus_20_n2 = (
        df_lmh_minus_20["nparticles: 2 - energy"].rolling(window=20).mean()
    )
    min_energy_lmh_minus_20_n2 = (
        df_lmh_minus_20["nparticles: 2 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_lmh_minus_20_n2 = (
        df_lmh_minus_20["nparticles: 2 - energy__MAX"].rolling(window=20).mean()
    )

    energy_lmh_minus_20_n4 = (
        df_lmh_minus_20["nparticles: 4 - energy"].rolling(window=20).mean()
    )
    min_energy_lmh_minus_20_n4 = (
        df_lmh_minus_20["nparticles: 4 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_lmh_minus_20_n4 = (
        df_lmh_minus_20["nparticles: 4 - energy__MAX"].rolling(window=20).mean()
    )

    energy_lmh_minus_20_n6 = (
        df_lmh_minus_20["nparticles: 6 - energy"].rolling(window=20).mean()
    )
    min_energy_lmh_minus_20_n6 = (
        df_lmh_minus_20["nparticles: 6 - energy__MIN"].rolling(window=20).mean()
    )
    max_energy_lmh_minus_20_n6 = (
        df_lmh_minus_20["nparticles: 6 - energy__MAX"].rolling(window=20).mean()
    )

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for V = 20
    axes[0].plot(steps_m_20, energy_m_20_n2, linestyle="--", color=palette[0], lw=2)
    # fill between
    axes[0].fill_between(steps_m_20, min_energy_m_20_n2, max_energy_m_20_n2, alpha=0.2)

    axes[0].plot(steps_lmh_20, energy_lmh_20_n2, linestyle="-", color=palette[0], lw=2)
    # fill between
    axes[0].fill_between(
        steps_lmh_20, min_energy_lmh_20_n2, max_energy_lmh_20_n2, alpha=0.2
    )

    axes[0].plot(steps_m_20, energy_m_20_n4, linestyle="--", color=palette[1], lw=2)
    # fill between
    axes[0].fill_between(steps_m_20, min_energy_m_20_n4, max_energy_m_20_n4, alpha=0.2)

    axes[0].plot(steps_lmh_20, energy_lmh_20_n4, linestyle="-", color=palette[1], lw=2)
    # fill between
    axes[0].fill_between(
        steps_lmh_20, min_energy_lmh_20_n4, max_energy_lmh_20_n4, alpha=0.2
    )

    axes[0].plot(steps_m_20, energy_m_20_n6, linestyle="--", color=palette[2], lw=2)
    # fill between
    axes[0].fill_between(steps_m_20, min_energy_m_20_n6, max_energy_m_20_n6, alpha=0.2)

    axes[0].plot(steps_lmh_20, energy_lmh_20_n6, linestyle="-", color=palette[2], lw=2)
    # fill between
    axes[0].fill_between(
        steps_lmh_20, min_energy_lmh_20_n6, max_energy_lmh_20_n6, alpha=0.2
    )

    # axes[0].set_title('Energy vs. Epoch for V = 20')
    axes[0].set_title("$V_0 = 20$")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Energy")
    axes[0].grid(True)

    # Plot for V = -20
    axes[1].plot(
        steps_m_minus_20, energy_m_minus_20_n2, linestyle="--", color=palette[0], lw=2
    )
    # fill between
    axes[1].fill_between(
        steps_m_minus_20, min_energy_m_minus_20_n2, max_energy_m_minus_20_n2, alpha=0.2
    )

    axes[1].plot(
        steps_lmh_minus_20,
        energy_lmh_minus_20_n2,
        linestyle="-",
        color=palette[0],
        lw=2,
    )
    # fill between
    axes[1].fill_between(
        steps_lmh_minus_20,
        min_energy_lmh_minus_20_n2,
        max_energy_lmh_minus_20_n2,
        alpha=0.2,
    )

    axes[1].plot(
        steps_m_minus_20, energy_m_minus_20_n4, linestyle="--", color=palette[1], lw=2
    )
    # fill between
    axes[1].fill_between(
        steps_m_minus_20, min_energy_m_minus_20_n4, max_energy_m_minus_20_n4, alpha=0.2
    )

    axes[1].plot(
        steps_lmh_minus_20,
        energy_lmh_minus_20_n4,
        linestyle="-",
        color=palette[1],
        lw=2,
    )
    # fill between
    axes[1].fill_between(
        steps_lmh_minus_20,
        min_energy_lmh_minus_20_n4,
        max_energy_lmh_minus_20_n4,
        alpha=0.2,
    )

    axes[1].plot(
        steps_m_minus_20, energy_m_minus_20_n6, linestyle="--", color=palette[2], lw=2
    )
    # fill between
    axes[1].fill_between(
        steps_m_minus_20, min_energy_m_minus_20_n6, max_energy_m_minus_20_n6, alpha=0.2
    )

    axes[1].plot(
        steps_lmh_minus_20,
        energy_lmh_minus_20_n6,
        linestyle="-",
        color=palette[2],
        lw=2,
    )
    # fill between
    axes[1].fill_between(
        steps_lmh_minus_20,
        min_energy_lmh_minus_20_n6,
        max_energy_lmh_minus_20_n6,
        alpha=0.2,
    )

    # axes[1].set_title('Energy vs. Epoch for V = -20')
    axes[1].set_title("$V_0 = - 20$")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Energy")
    axes[1].grid(True)

    # Add combined legend
    custom_lines = [
        plt.Line2D([0], [0], color=palette[0], lw=2),
        plt.Line2D([0], [0], color=palette[1], lw=2),
        plt.Line2D([0], [0], color=palette[2], lw=2),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="--"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="-"),
    ]

    fig.legend(
        custom_lines,
        ["N = 2", "N = 4", "N = 6", "Metro", "LangevinMetro"],
        loc="upper center",
        ncol=5,
        frameon=True,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.9),
    )

    # tight
    plt.tight_layout()

    plot_style.save("fermion_polarized/lmh_vs_m")


if __name__ == "__main__":
    # energy_convergence_loop(v0=-20)
    # energy_convergence_loop(v0=-10)
    # energy_convergence_loop(v0=10)
    # energy_convergence_loop(v0=20)
    # plot_opti_sweep()
    # plot_obdm_histogram([2, 3], [-20, 0], "VMC")
    # v0_values = [-20, -10, 10, 20]
    # energy_convergence_loop_side_by_side(v0_values)
    # plot_lmh_vs_m()
    # energy_convergence_loop_side_by_side_jastrow(v0_values)
    # plot_density_profile([2,3,4,5,6], [-20, -10, 0, 10, 20], "DSFFN")
    # plot_density_profile([2], [-20, -10, 0, 10, 20], "all")
    # plot_density_profile([3], [-20, -10, 0, 10, 20], "all")
    # plot_density_profile([4], [-20, -10, 0, 10, 20], "all")
    # plot_density_profile([5], [-20, -10, 0, 10, 20], "all")
    # plot_density_profile([6], [-20, -10, 0, 10, 20], "all")
    # plot_lmh_vs_m()
    pass
