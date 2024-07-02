import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter

from nqs.state.utils import plot_style


DATA_PATH = "../data/fermion_dots"
FIG_PATH = "figs/fermion_dots"
import pandas as pd
import glob

from matplotlib.ticker import FuncFormatter, MaxNLocator

import fitz  # PyMuPDF


def scientific_notation(z, pos):
    return f"{z:.1e}"


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


def energy_convergence_loop_side_by_side(n_particles, omega, interaction=0):
    """
    Plot the convergence of the energy as a function of the number of samples for multiple v0 values.
    """
    z = 1.96  # 95% confidence interval

    # Set the seaborn pastel palette
    sns.set_palette("pastel")
    palette = sns.color_palette()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    omega_values = omega

    # Define color map for different nqs_types
    color_map = {}

    for omega in omega_values:
        df = pd.read_csv(DATA_PATH + f"/energies_and_stds_omega_{omega}.csv")
        nqs_types = df["nqs_type"].unique()
        print(nqs_types)
        color_map.update(
            {
                nqs_type: palette[i % len(palette)]
                for i, nqs_type in enumerate(nqs_types)
            }
        )

    for ax, omega in zip(axs, omega_values):
        print(ax)
        for n_particles in n_particles_list:
            df = pd.read_csv(DATA_PATH + f"/energies_and_stds_omega_{omega}.csv")

            nqs_types = df["nqs_type"].unique()

            for nqs_type in nqs_types:
                # Filter data for the current n_particles and nqs_type
                filtered_df = df[
                    (df["n_particles"] == n_particles) & (df["nqs_type"] == nqs_type)
                ]
                if filtered_df.empty:
                    print(f"empty df for {nqs_type} and {n_particles}")
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
                ax.fill_between(
                    energies.index, stds_minus, stds_plus, alpha=0.1, color="purple"
                )

                # Plot energies
                ax.plot(energies, label=f"{nqs_type}", color=color_map[nqs_type])

            # Add ground state line
            energies = {"2": omega * 2, "6": 10 * omega, "12": 28 * omega}

            if interaction == 0:
                analytical = energies[str(n_particles)]
                ax.axhline(
                    y=analytical,
                    color="black",
                    linestyle="--",
                    label="Analytical",
                    lw=1,
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
                    df_hf = df_hf[df_hf["N"] == n_particles]
                    df_hf = df_hf[df_hf["omega"] == omega]
                    e_hf = df_hf["Energy"].values[0]
                    ax.axhline(y=e_hf, color="black", linestyle="-.", label="HF", lw=1)

                except Exception as e:
                    raise ValueError(f"{e} No data for HF")
                try:
                    df_ci = pd.read_csv(DATA_PATH + "/data_ci.csv")
                    df_ci = df_ci[df_ci["N"] == n_particles]
                    df_ci = df_ci[df_ci["omega"] == omega]
                    e_ci = df_ci["Energy"].values[0]
                    ax.axhline(y=e_ci, color="black", linestyle=":", label="CI", lw=1)
                except Exception as e:
                    raise ValueError(f"{e} No data for CI")

        # Add omega value to the top center of the plot
        ax.text(
            0.5,
            0.95,
            f"$\omega$ = {omega}",  # noqa
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=18,
        )
        # only write epochs on the bottom plots
        if ax in axs[2:]:
            ax.set_xlabel("Epochs")

        # set axis to log scale
        ax.set_yscale("log")
        ax.set_ylabel("$\ln$(Energy)")  # noqa

    # Custom legend to include markers for n_particles
    handles, labels = axs[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    legend_labels = list(unique_labels.keys())
    legend_handles = [
        unique_labels[label] for label in legend_labels if label in unique_labels
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        fontsize=18,
        bbox_to_anchor=(0.5, 1.01),
        markerscale=1,
        borderpad=0.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_style.save("fermion_dots/E_conv_ferm_dots_NI")


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
            stds = z * filtered_df["std_error"] / np.sqrt(filtered_df["batch_size"])

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
                alpha=0.1,
                color=color_map[correlation],
            )

            # Plot energies
            ax.plot(energies, label=f"{correlation}", color=color_map[correlation])

        # Add ground state line
        dimensions = 2
        if v0 == 0:
            analytical = dimensions * float(omega) * 0.5 * n_particles**2
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


def energy_convergence_loop(omega, n_particles=2, interaction=0):
    """
    Plot the convergence of the energy as a function of the number of samples.
    """
    # Load the data
    df = pd.read_csv(DATA_PATH + f"/energies_and_stds_omega_{omega}.0.csv")

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

        energies = {
            "2": omega * 2,
            "6": 12 * omega,
            "12": omega * 2 + omega * (2) * 4 + omega * (3) * 6,
        }
        if interaction == 0:
            analytical = energies[str(n_particles)]
            print(analytical)
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
                df_hf = df_hf[df_hf["N"] == n_particles]
                df_hf = df_hf[df_hf["omega"] == omega]
                e_hf = df_hf["Energy"].values[0]
                plt.axhline(y=e_hf, color="black", linestyle="-.", label="HF", lw=1)

            except Exception as e:
                raise ValueError(f"{e} No data for HF")
            try:
                df_ci = pd.read_csv(DATA_PATH + "/data_ci.csv")
                df_ci = df_ci[df_ci["N"] == n_particles]
                df_ci = df_ci[df_ci["omega"] == omega]
                e_ci = df_ci["Energy"].values[0]
                plt.axhline(y=e_ci, color="black", linestyle=":", label="CI", lw=1)
            except Exception as e:
                raise ValueError(f"{e} No data for CI")

        # Add V_0 to the top center of the plot
        plt.text(
            0.5,
            0.95,
            f"$\omega$ = {omega}",  # noqa
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
    plot_style.save(f"fermion_polarized/energy_convergence_omega_{omega}")


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


def plot_density_profile(n_values, v0_values, nqs):
    # line types for different NQS types
    line_types = {"VMC": "-", "RBM": "--", "DSFFN": "-."}

    # sns pastel palette
    colors = {
        "VMC": sns.color_palette("pastel")[0],
        "RBM": sns.color_palette("pastel")[1],
        "DSFFN": sns.color_palette("pastel")[2],
    }

    def plot_position_density(positions, omega_value, n_value, nqs, ax, fontsize=14):
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
            ax.set_title(f"$V_0 = {omega_value}$")

        if n_value == max(n_values):
            ax.set_xlabel("Position, x")

        if omega_value == 0 and (n_value == 3 or n_value == 2):
            ax.legend()
        if omega_value == min(v0_values):
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

    if nqs == "all":
        file_base_pattern_vmc = (
            f"{DATA_PATH}/sampler/N{{}}_omega{{}}_energies_and_pos_VMC_ch*.h5"
        )
        file_base_pattern_rbm = (
            f"{DATA_PATH}/sampler/N{{}}_omega{{}}_energies_and_pos_RBM_ch*.h5"
        )
        file_base_pattern_ds = (
            f"{DATA_PATH}/sampler/N{{}}_omega{{}}_energies_and_pos_DSFFN_ch*.h5"
        )
        file_base_patterns = [
            file_base_pattern_vmc,
            file_base_pattern_rbm,
            file_base_pattern_ds,
        ]
    else:
        file_base_patterns = [
            f"{DATA_PATH}/sampler/N{{}}_omega{{}}_energies_and_pos_{nqs}_ch*.h5"
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
                    print(f"Pattern: {file_pattern}")
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

    # plt.tight_layout()
    # plot_style.save(f"fermion_polarized/density_profile_N{n_values}_nqs_{nqs}")


def plot_densities(n_values, omega_values, nqs):
    def plot_two_body_density(positions, omega, n_bins=100, max_distance=None):
        n_samples, n_particles_2d = positions.shape
        n_particles = n_particles_2d // 2

        # Reshape to separate particles and dimensions
        positions_reshaped = positions.reshape(n_samples, n_particles, 2)

        if max_distance is None:
            max_distance = np.max(np.linalg.norm(positions_reshaped, axis=2))

        # Create bins for radial distances
        bins = np.linspace(0, max_distance, n_bins + 1)

        # Initialize the 2D histogram
        rho_2 = np.zeros((n_bins, n_bins))

        # Calculate two-body density
        for i in range(n_particles):
            r1 = np.linalg.norm(positions_reshaped[:, i, :], axis=1)
            for j in range(i + 1, n_particles):
                r2 = np.linalg.norm(positions_reshaped[:, j, :], axis=1)

                # Use r1 and r2 to bin the data
                r1_indices = np.digitize(r1, bins) - 1
                r2_indices = np.digitize(r2, bins) - 1

                for r1_idx, r2_idx in zip(r1_indices, r2_indices):
                    if r1_idx < n_bins and r2_idx < n_bins:
                        rho_2[r1_idx, r2_idx] += 1

        # Apply area correction
        bin_centers = (bins[1:] + bins[:-1]) / 2
        bin_widths = np.diff(bins)
        area_correction = 1 / (
            2
            * np.pi
            * bin_centers[:, np.newaxis]
            * bin_centers[np.newaxis, :]
            * bin_widths[:, np.newaxis]
            * bin_widths[np.newaxis, :]
        )
        rho_2 *= area_correction

        # Normalize
        rho_2 /= np.sum(rho_2)
        rho_2 *= n_particles * (n_particles - 1)

        # now copy paste this in the other quadrants to get the negative rs too
        # Create the full matrix by mirroring the quadrants
        full_rho_2 = np.zeros((2 * n_bins, 2 * n_bins))
        full_rho_2[:n_bins, :n_bins] = np.flip(rho_2, axis=1)
        full_rho_2[:n_bins, n_bins:] = rho_2
        full_rho_2[n_bins:, :] = full_rho_2[:n_bins, :]

        full_rho_2 = np.vstack(
            (np.flip(full_rho_2[:n_bins], axis=0), full_rho_2[n_bins:])
        )

        # pass gaussian filter to smooth the plot
        full_rho_2 = gaussian_filter(full_rho_2, sigma=1)

        # Plot the two-body density
        fig, ax = plt.subplots(figsize=(5, 5))
        # elliminate grid
        ax.grid(False)

        if omega_values[0] == 0.1:
            max_distance = 12
        elif omega_values[0] == 0.5:
            max_distance = 6
        elif omega_values[0] == 1.0:
            max_distance = 4
        elif omega_values[0] == 0.28:
            max_distance = 8

        extent = [-max_distance, max_distance, -max_distance, max_distance]
        im = ax.imshow(
            full_rho_2, extent=extent, origin="lower", cmap="BuPu", aspect="auto"
        )

        # chose font size for colorbar
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(r"$n(\mathbf{r}_1, \mathbf{r}_2)$", fontsize=20)
        # set scientific notation for colorbar
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        # Set the labels

        # add text to the top left corner
        ax.text(
            0.05, 0.95, f"N = {n_values[0]}, {nqs}", transform=ax.transAxes, fontsize=18
        )

        ax.set_xlabel("$\mathbf{r}_i$", fontsize=20)  # noqa
        ax.set_ylabel("$\mathbf{r}_j$", fontsize=20)  # noqa
        print(f"Plotting two-body density for N = {n_values[0]}, omega = {omega}")
        plot_style.save(f"fermion_dots/two_body_density_N{n_values}_nqs_{nqs}_{omega}")

    def plot_3d_density(positions, ax):
        dim = 2
        positions = positions.reshape(-1, dim)
        x_min, y_min = positions.min(axis=0)
        x_max, y_max = positions.max(axis=0)

        x_min_abs, x_max_abs = np.abs(x_min), np.abs(x_max)
        y_min_abs, y_max_abs = np.abs(y_min), np.abs(y_max)

        x_min, x_max = -max(x_min_abs, x_max_abs), max(x_min_abs, x_max_abs)
        y_min, y_max = -max(y_min_abs, y_max_abs), max(y_min_abs, y_max_abs)

        bins = 500
        x_edges = np.linspace(x_min, x_max, bins + 1)
        y_edges = np.linspace(y_min, y_max, bins + 1)

        hist, x_edges, y_edges = np.histogram2d(
            positions[:, 0], positions[:, 1], bins=(x_edges, y_edges), density=True
        )

        sigma = [5, 5]
        Z = gaussian_filter(hist, sigma)

        X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
        X, Y = X.T, Y.T

        # make the lines in the surface plot be black
        ax.plot_surface(X, Y, Z, cmap="BuPu_r", linewidth=0, color="black", alpha=0.7)
        ax.plot_wireframe(X, Y, Z, color="grey", linewidth=0.5, alpha=0.5)

        y_line = y_edges[:-1]
        z_y_line = Z[Z.shape[0] // 2, :]
        x_line = x_edges[:-1]
        z_x_line = Z[:, Z.shape[1] // 2]
        ax.plot(
            x_line,
            y_max * np.ones_like(x_line),
            z_x_line,
            color="grey",
            linestyle="-",
            linewidth=1.5,
        )
        ax.plot(
            x_min * np.ones_like(y_line),
            y_line,
            z_y_line,
            color="grey",
            linestyle="-",
            linewidth=1.5,
        )

        # add title
        ax.set_xlabel("X", fontsize=15)
        ax.set_ylabel("Y", fontsize=15)
        ax.set_zlabel("$n(\mathbf{r}_1, \mathbf{r}_2)$", fontsize=20)  # noqa

        # remove the grid
        ax.grid(False)

        # make z scale be scientific with format_e function
        # make z axis to the left
        ax.zaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.tick_params(axis="both", which="minor", labelsize=16)
        ax.zaxis.set_major_formatter(FuncFormatter(scientific_notation))
        for label in ax.zaxis.get_ticklabels():
            label.set_rotation(0)  # Set the desired angle
            label.set_horizontalalignment("right")  # Align labels to the right

        # Set font size and padding for Z axis labels
        # ax.tick_params(axis='z', which='major', labelsize=10, pad=15)
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))  # Set a smaller number of bins

        ax.text2D(
            0.05, 0.95, f"N = {n_values[0]}, {nqs}", transform=ax.transAxes, fontsize=22
        )

    if nqs == "all":
        file_base_patterns = [
            f"{DATA_PATH}/sampler/N{{}}_omega{{}}_energies_and_pos_{type}_ch*.h5"
            for type in ["VMC", "RBM", "DSFFN"]
        ]
    else:
        file_base_patterns = [
            f"{DATA_PATH}/sampler/N{{}}_omega{{}}_energies_and_pos_{nqs}_ch*.h5"
        ]

    if n_values[0] == 2:
        figsize = (12, 8)

    else:
        figsize = (10, 3)

    fig, axes = plt.subplots(
        len(n_values), len(omega_values), figsize=figsize, squeeze=False
    )

    fig_3d, ax_3d = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))

    ####### 3D DENSITY PLOT ######## # noqa

    n = n_values[0]
    omega = omega_values[0]
    # assert int(omega) == 1 , "Only omega = 1 is supported for 3D density plot"
    for n in n_values:
        all_positions = []
        for file_base_pattern in file_base_patterns:
            file_pattern = file_base_pattern.format(n, float(omega))
            file_paths = glob.glob(file_pattern)
            if file_paths:
                for file_path in file_paths:
                    with h5py.File(file_path, "r") as hdf:
                        positions = np.array(hdf["positions"])
                        all_positions.append(positions)

        if all_positions:
            # print(f"Plotting 3D density profile for N = {n}")
            combined_positions = np.vstack(all_positions)

            # plot_3d_density(combined_positions, ax_3d)
            plot_two_body_density(combined_positions, omega)

        plt.tight_layout()

        plot_style.save(f"fermion_dots/density_profile_3d_N{n}_nqs_{nqs}_{omega}")

        crop_pdf(
            "/Users/orpheus/Documents/Masters/NeuralQuantumState/analysis/figs/"
            + f"fermion_dots/density_profile_3d_N{n}_nqs_{nqs}_{omega}.pdf",
            "/Users/orpheus/Documents/Masters/NeuralQuantumState/analysis/figs/"
            + f"fermion_dots/density_profile_3d_N{n}_nqs_{nqs}_{omega}_c.pdf",
            1.5,
            0,
            1,
            0,
        )
        # remove the old one and rename the new one to the old one
        os.remove(
            "/Users/orpheus/Documents/Masters/NeuralQuantumState/analysis/figs/"
            + f"fermion_dots/density_profile_3d_N{n}_nqs_{nqs}_{omega}.pdf"
        )
        os.rename(
            "/Users/orpheus/Documents/Masters/NeuralQuantumState/analysis/figs/"
            + f"fermion_dots/density_profile_3d_N{n}_nqs_{nqs}_{omega}_c.pdf",
            "/Users/orpheus/Documents/Masters/NeuralQuantumState/analysis/figs/"
            + f"fermion_dots/density_profile_3d_N{n}_nqs_{nqs}_{omega}.pdf",
        )


def plt_radial_profile(n_values, omega_values, nqs):
    line_types = {"VMC": "-", "RBM": "--", "DSFFN": "-."}

    colors = {
        "VMC": sns.color_palette("pastel")[0],
        "RBM": sns.color_palette("pastel")[1],
        "DSFFN": sns.color_palette("pastel")[2],
    }

    if nqs == "all":
        nqs_types = ["VMC", "RBM", "DSFFN"]
        file_base_patterns = [
            f"{DATA_PATH}/sampler/N{{}}_omega{{}}_energies_and_pos_{type}_ch*.h5"
            for type in nqs_types
        ]
    else:
        nqs_types = [nqs]
        file_base_patterns = [
            f"{DATA_PATH}/sampler/N{{}}_omega{{}}_energies_and_pos_{nqs}_ch*.h5"
        ]

    if n_values[0] == 2:
        figsize = (10, 3.3)
    else:
        figsize = (10, 3)
    fontsize = 14
    # else:
    #    figsize = (10, 3)
    #    fontsize = 14

    fig, axes = plt.subplots(
        len(n_values), len(omega_values), figsize=figsize, squeeze=False
    )

    def plot_radial_density(positions, omega_value, n_value, nqs, ax, fontsize=14):
        # Calculate radial distances
        positions = positions.reshape(-1, n_value, 2)
        positions = np.linalg.norm(positions, axis=2)
        positions = positions.flatten()

        n_bins = 120
        hist, bin_edges = np.histogram(positions, bins=n_bins, density=False)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_widths = np.diff(bin_edges)

        correction_factor = 1 / (2 * np.pi * bin_centers * bin_widths)
        corrected_hist = hist * correction_factor

        corrected_hist = (
            n_value * corrected_hist / (np.sum(corrected_hist) * bin_widths)
        )

        ax.plot(
            bin_centers,
            corrected_hist,
            label=nqs,
            color=colors[nqs],
            linestyle=line_types[nqs],
        )

        if n_value == 2:
            ax.set_title(f"$\omega_0 = {omega_value}$")  # noqa

        if n_value == 20:
            ax.set_xlabel("Radial Distance, r")

        if omega_value == 1.0:
            ax.legend()
            ax.set_ylabel("$n(r)$")

        if omega_value == min(omega_values):
            ax.text(
                0.95,
                0.95,
                f"N = {n_value}",
                horizontalalignment="right",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=fontsize,
            )

        ax.set_ylim(0, np.max(corrected_hist) * 1.1)
        ax.set_xlim(0, np.max(bin_edges) * 1.1)

    for i, n in enumerate(n_values):
        for j, omega in enumerate(omega_values):
            all_positions = {nqs_type: [] for nqs_type in nqs_types}
            for file_base_pattern in file_base_patterns:
                nqs_type = file_base_pattern.split("_")[-2]
                file_pattern = file_base_pattern.format(n, float(omega))
                file_paths = glob.glob(file_pattern)

                if not file_paths:
                    print(
                        f"No files found for N = {n}, omega = {omega}, nqs = {nqs_type}"
                    )
                    continue

                for file_path in file_paths:
                    with h5py.File(file_path, "r") as hdf:
                        try:
                            positions = np.array(hdf["positions"])
                            all_positions[nqs_type].append(positions)
                        except Exception as e:
                            print(f"Error reading positions from {file_path}: {e}")

                print(f"Found {len(file_paths)} files for N = {n}, omega = {omega}")

            for nqs_type in nqs_types:
                if all_positions[nqs_type]:
                    combined_positions = np.vstack(all_positions[nqs_type])
                    plot_radial_density(
                        combined_positions,
                        omega,
                        n,
                        nqs_type,
                        axes[i, j],
                        fontsize=fontsize,
                    )

    plt.tight_layout()
    plot_style.save(f"fermion_dots/radial_profile_N{n_values}_nqs_{nqs}")


def inches_to_points(inches):
    return inches * 72


def crop_pdf(
    input_pdf_path, output_pdf_path, crop_left, crop_top, crop_right, crop_bottom
):
    # Convert inches to points
    crop_left = inches_to_points(crop_left)
    crop_top = inches_to_points(crop_top)
    crop_right = inches_to_points(crop_right)
    crop_bottom = inches_to_points(crop_bottom)

    # Open the input PDF
    pdf_document = fitz.open(input_pdf_path)

    # Iterate through each page and crop it
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        # Get the page dimensions
        rect = page.rect
        # Define the cropping rectangle in points
        crop_rect = fitz.Rect(
            rect.x0 + crop_left,
            rect.y0 + crop_top,
            rect.x1 - crop_right,
            rect.y1 - crop_bottom,
        )
        page.set_cropbox(crop_rect)

    # Save the cropped PDF to the output path
    pdf_document.save(output_pdf_path)
    print(f"Cropped PDF saved to {output_pdf_path}")


if __name__ == "__main__":
    # energy_convergence_loop(omega=1, n_particles=[2,6])
    # energy_convergence_loop(v0=-10)
    # energy_convergence_loop(v0=10)
    # energy_convergence_loop(v0=20)
    # plot_opti_sweep()
    # plot_obdm_histogram([2, 3], [-20, 0], "VMC")
    omega_values = [0.5]
    n_particles_list = [20]
    nqs_types = ["DSFFN"]

    for nqs in nqs_types:
        for n_particles in n_particles_list:
            for omega in omega_values:
                # plot_density_profile([n_particles], [omega], nqs)
                plot_densities([n_particles], [omega], nqs)

    # plt_radial_profile(n_particles_list, omega_values, "VMC")
    # plt_radial_profile([2], [1.0, 0.5, 0.28, 0.1], 'all')
    # plt_radial_profile([6], [1.0, 0.5, 0.28, 0.1], 'all')
    # plt_radial_profile([12], [1.0, 0.5, 0.28, 0.1], 'all')
    # plt_radial_profile([20], [1.0, 0.5, 0.28, 0.1], 'all')

    # energy_convergence_loop_side_by_side(n_particles_list, omega_values, interaction=0)
    # energy_convergence_loop_side_by_side_jastrow(omega_values)
    pass
