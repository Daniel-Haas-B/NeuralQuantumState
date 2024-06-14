# ! ##########
# DISCLAIMER: this material is not mine. It was taken from https://github.com/arnaurios/1D_fermions/blob/master/hartree_fock/hartree_fock_1Dfermions.ipynb
# ! ##########
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "/Users/orpheus/Documents/Masters/NeuralQuantumState/data/fermion_polarized"

import math
import pandas as pd


# WAVE FUNCTIONS OF THE 1D HARMONIC OSCILLATOR
def wfho(n, x):
    if n > 20:
        print("N value of Harmonic Oscillator Wavefunction too large: exiting code")
        exit()

    nfac = math.factorial(n)
    pi = math.pi
    norm = 1.0 / np.sqrt(np.power(2, n) * nfac) / np.power(pi, 0.25)

    if n == 0:
        Hn = np.ones_like(x)
    elif n == 1:
        Hn = 2.0 * x
    elif n > 1:
        hm2 = np.ones_like(x)
        hm1 = 2.0 * x
        for m in range(2, n + 1):
            hmx = 2.0 * x * hm1 - 2.0 * np.real(m - 1) * hm2
            hm2 = hm1
            hm1 = hmx
        Hn = hmx
    else:
        print("n is wrong! Exiting code")
        exit()

    wf = norm * np.exp(-np.power(x, 2) / 2.0) * Hn

    return wf


def save_data(
    Astr, V0string, s_string, enerhf, enerhfp, ekin0hf, epot0hf, esum0hf, diffs
):
    data = {
        "A": Astr,
        "V0": V0string,
        "s": s_string,
        "Energy": enerhf,
        "Energy_P": enerhfp,
        "Kinetic": ekin0hf,
        "Potential": epot0hf,
        "Sum": esum0hf,
        "Diff": diffs,
    }

    df = pd.DataFrame(data, index=[0])

    if not os.path.exists(DATA_PATH + "/data_hf.csv"):
        df.to_csv(DATA_PATH + "/data_hf.csv", mode="a", header=True)
    else:
        # check if there is not already the same data
        data_hf = pd.read_csv(DATA_PATH + "/data_hf.csv")
        # just needs to check if there is any energy value that is the same

        if not (data_hf["Energy"] == enerhf).any():
            df.to_csv(DATA_PATH + "/data_hf.csv", mode="a", header=False)
        else:
            print("Data already exists")


def main(A=2, V0=-20, s=0.5):
    pi = math.pi
    zi = complex(0.0, 1.0)

    eps_system = sys.float_info.epsilon
    zero_low = eps_system * 1000  # noqa

    # NUMBER OF PARTICLES
    A_num_part = A
    Astr = str(A_num_part)
    print(f"using {Astr} particles")

    # DEFINES THE VALUES OF INTERACTION STRENGTH AND INTERACTION RANGE
    # nV=41
    nV = 1
    nS = 1
    V_strength = np.linspace(V0, V0, nV)
    S_range = np.linspace(s, s, nS)
    VV, ss = np.meshgrid(V_strength, S_range)

    if A_num_part < 5:
        xL = 5.0
        Nx = 200
    else:
        xL = 6.0
        Nx = 240

    # GRID SPACING
    delx = 2 * xL / Nx

    # MESH IN X-SPACE - FROM -xL+del x UP TO +xL
    xx = np.zeros(Nx)
    xx = delx * (np.arange(Nx) - Nx / 2.0)
    [x1, x2] = np.meshgrid(xx, xx)

    # SPACING IN MOMENTUM SPACE
    delp = 2.0 * pi / (2.0 * xL)

    # MESH IN p SPACE
    pp = np.zeros(Nx)
    for i in range(0, Nx):
        pp[i] = (i - Nx / 2) * delp

    # SECOND DERIVATIVE MATRIX
    cder2 = np.zeros((Nx, Nx), complex)
    der2 = np.zeros((Nx, Nx))
    # LOOP OVER I AND J
    for i, xi in enumerate(xx):
        for j, xj in enumerate(xx):
            cder2[i, j] = np.dot(np.exp(zi * (xj - xi) * pp), np.power(pp, 2))
            # cder2[i,j] = np.dot( np.cos( (xj-xi)*pp )+zi*np.sin( (xj-xi)*pp ), np.power(pp,2) )

    # ADD PHYSICAL FACTORS AND KEEP REAL PART ONLY FOR SECOND DERIVATIVE
    der2 = -np.real(cder2) * delx * delp / 2.0 / pi
    kin_mat = -der2 / 2.0  # COULD ADD HBAR2/M HERE IF OTHER UNITS USED

    # HARMONIC OSCILLATOR MATRIX IN REAL SPACE - DIAGONAL
    U_HO = np.power(xx, 2.0) / 2.0

    # HARTREE-FOCK POTENTIAL
    accu = 1e-9
    itermax = 20000
    pfac = 1.0 / np.abs(np.amax(kin_mat))
    ffac = 0.4

    # PREPARE ARRAYS
    hf_den = np.zeros(Nx)
    denf = np.zeros(Nx)
    hf_den_mat = np.zeros((Nx, Nx))
    H = np.zeros((Nx, Nx))

    # INITIALIZE ARRAYS
    # NOTE Nmax IS USED AS MAXIMUM IN PYTHON RANGE ARRAYS, SO IT IS ACTUALLY A-1 IN MATHS TERMS
    Nmax = A_num_part
    wfy = np.zeros((Nx, Nmax))
    spe = np.zeros(Nmax)
    for ieig in range(Nmax):
        wfy[:, ieig] = wfho(ieig, xx)

    # DEFINE MATRICES AS A FUNCTION OF S AND V
    energy = np.zeros((5, nV, nS))
    A_num_sum = np.zeros((nV, nS))
    rms_den = np.zeros((nV, nS))

    header_screen = (
        "# ITER".ljust(8)
        + "NUM_PART".ljust(14)
        + "X_CM".ljust(13)
        + "EHF".ljust(13)
        + "EHF2".ljust(13)
        + "EKIN".ljust(13)
        + "EPOT".ljust(13)
        + "ESUM".ljust(13)
        + "DIFFS"
    )

    # LOOP OVER INTERACTION RANGE
    for iS, s in enumerate(S_range):
        s_string = "{:.1f}".format(s)
        # LOOP OVER INTERACTION STRENGTH
        for iV, V0 in enumerate(V_strength):
            V0string = "{:.1f}".format(V0)
            print("\ns_range=" + s_string + " V0=", V0string)
            print(header_screen)

            # INTERACTION POTENTIAL MATRIX
            Vint = (
                V0
                / np.sqrt(2.0 * pi)
                / s
                * np.exp(-np.power(x1 - x2, 2) / 2.0 / np.power(s, 2))
            )

            # START HARTREE-FOCK ITERATION PROCEDURE
            iter = 0
            diffs = 10.0
            while diffs > accu and iter < itermax:
                iter = iter + 1
                # ... PREPARE DENSITY AND DENSITY MATRIX FROM ORBITALS
                hf_den = 0.0
                hf_den_mat = 0.0
                for ieig in range(Nmax):
                    hf_den = hf_den + np.power(abs(wfy[:, ieig]), 2)
                    hf_den_mat = hf_den_mat + np.outer(wfy[:, ieig], wfy[:, ieig])

                # COMPUTE DENSITY
                denf = hf_den

                # MEAN-FIELD - DIRECT TERM
                Udir = delx * np.matmul(Vint, denf)

                # MEAN-FIELD - DIRECT TERM
                Umf_mat = -delx * Vint * hf_den_mat

                # ADD ALL MEAN-FIELD TERMS TOGETHER
                np.fill_diagonal(Umf_mat, Umf_mat.diagonal() + Udir + U_HO)
                # MEAN-FIELD ALONG DIAGONAL

                # HAMILTONIAN Nx x Nx MATRIX
                H = kin_mat + Umf_mat

                # cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                # ... USE RAYLEIGH APPROXIMATION TO FIND APPROXIMATE EIGENVALUES
                # ... LOOP OVER EIGENVALUES
                diffs = 0.0
                ekin0hf = 0.0
                epot0hf = 0.0
                for ieig in range(Nmax):
                    # ... GUESS EIGENVALUE WITH RAYLEIGH METHOD
                    wf0 = wfy[:, ieig]
                    wff = np.matmul(H, wf0)
                    spe[ieig] = np.real(np.dot(wf0, wff)) * delx

                    # GRADIENT ITERATION
                    wfb = wf0 - wff * pfac
                    norm = math.fsum(np.power(np.abs(wfb), 2)) * delx
                    wff = wfb / np.sqrt(norm)
                    # print(wff[99])

                    # ... ORTHOGONALIZATION
                    wfo = 0.0
                    for jeig in range(0, ieig):
                        wfo = wfo + wfy[:, jeig] * np.dot(wfy[:, jeig], wff) * delx
                    wff = wff - wfo
                    norm = math.fsum(np.power(np.abs(wff), 2)) * delx
                    wff = wff / norm

                    # ... vec_k+1 = (A-mI)^-1 * vec_k
                    diffs = diffs + np.amax(abs(wf0 - wff))
                    wfy[:, ieig] = ffac * wff + (1.0 - ffac) * wf0

                    if ieig <= Nmax:
                        ekin0hf = (
                            ekin0hf
                            + np.real(
                                np.dot(wfy[:, ieig], np.matmul(kin_mat, wfy[:, ieig]))
                            )
                            * delx
                        )
                        epot0hf = (
                            epot0hf
                            + np.real(
                                np.dot(wfy[:, ieig], np.matmul(Umf_mat, wfy[:, ieig]))
                            )
                            * delx
                        )

                esum0hf = np.sum(spe[0:Nmax])

                # HARTREE-FOCK DENSITY IS USED TO COMPUTE AVERAGE POSITION xa AND RMS x2_av
                xa = np.sum(hf_den) * delx
                x2_av = delx * np.dot(hf_den, np.power(xx, 2)) / (delx * np.sum(hf_den))

                # ENERGY FROM GMK SUMRULE
                eho = np.sum(hf_den * U_HO) * delx / 2.0
                enerhfp = (esum0hf + ekin0hf) / 2.0 + eho

                # ENERGY FROM INTEGRAL (DF)
                epothf = epot0hf - 2.0 * eho
                enerhf = esum0hf - epothf / 2.0

                # PRINT DATA TO SCREEN
                formatd = ["%12.6E"] * 8
                formatd = ["%4i"] + formatd
                if iter % 1000 == 0:
                    ddd = np.vstack(
                        [
                            iter,
                            xa,
                            x2_av,
                            enerhf,
                            enerhfp,
                            ekin0hf,
                            epot0hf,
                            esum0hf,
                            diffs,
                        ]
                    )
                    np.savetxt(sys.stdout, ddd.T, fmt=formatd)

            # ITERATION LOOP IS OVER - PRINT FINAL RESULTS
            energy[0, iV, iS] = enerhf
            energy[1, iV, iS] = enerhfp
            energy[2, iV, iS] = ekin0hf
            energy[3, iV, iS] = eho
            energy[4, iV, iS] = epothf

            A_num_sum[iV, iS] = xa
            rms_den[iV, iS] = x2_av

            # SAVE DATA
            save_data(
                Astr,
                V0string,
                s_string,
                enerhf,
                enerhfp,
                ekin0hf,
                epot0hf,
                esum0hf,
                diffs,
            )


def plot_data(
    Astr, V0string, s_string, xx, denf, hf_den_mat, pair_dist, nS, nV, s, xL, Nx
):
    # CREATE DATA/PLOT DIRECTORY IF THEY DO NOT EXIST
    # PREPARE DATA AND PLOT FOLDERS
    folder_numerics = "xL" + str(xL) + "_Nx" + str(Nx)

    datafolder = "data"
    plotfolder = "plots"
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)

    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)

    if not os.path.exists("data/" + folder_numerics):
        os.makedirs("data/" + folder_numerics)

    if not os.path.exists("plots/" + folder_numerics):
        os.makedirs("plots/" + folder_numerics)
    plot_folder = "plots/" + folder_numerics + "/"

    ##############################################################################
    # PLOTS OF DENSITY
    if nS == 1:
        plot_filedd = (
            plot_folder + "density_" + Astr + "_particles_V0=" + V0string + ".pdf"
        )
    elif nV == 1:
        plot_filedd = (
            plot_folder + "density_" + Astr + "_particles_s=" + s_string + ".pdf"
        )
    else:
        plot_filedd = (
            plot_folder
            + "density_"
            + Astr
            + "_particles_V0="
            + V0string
            + "_s="
            + s_string
            + ".pdf"
        )

    if os.path.exists(plot_filedd):
        os.remove(plot_filedd)

    plt.xlabel("Distance, x [ho units]")
    plt.ylabel("Density, $n(r)$")
    plt.title("Density, A=" + Astr + ", $V_0=$" + V0string + ", $s=$" + str(s))
    plt.plot(xx, denf)
    plt.show()
    plt.savefig(plot_filedd)
    plt.close()

    ##############################################################################
    # PLOTS OF DENSITY MATRIX
    if nS == 1:
        plot_filedm = (
            plot_folder + "denmat_" + Astr + "_particles_V0=" + V0string + ".pdf"
        )
    elif nV == 1:
        plot_filedm = (
            plot_folder + "denmat_" + Astr + "_particles_s=" + s_string + ".pdf"
        )
    else:
        plot_filedm = (
            plot_folder
            + "denmat_"
            + Astr
            + "_particles_V0="
            + V0string
            + "_s="
            + s_string
            + ".pdf"
        )

    if os.path.exists(plot_filedm):
        os.remove(plot_filedm)

    fig, ax = plt.subplots()
    fcont = ax.contourf(xx, xx, hf_den_mat, cmap="coolwarm")
    ax.contour(xx, xx, hf_den_mat, colors="k")
    ax.axis("square")
    fig.colorbar(fcont, ax=ax)
    ax.set_xlabel("Position, $x_1$ [ho units]")
    ax.set_ylabel("Position, $x_2$ [ho units]")
    ax.set_title(
        "Density matrix, A=" + Astr + ", $V_0=$" + V0string + ", $s=$" + str(s)
    )
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    plt.savefig(plot_filedm)
    plt.show()
    plt.close(fig)

    ##############################################################################
    # PLOTS OF PAIR DISTRIBUTION
    if nS == 1:
        plot_filedm = (
            plot_folder + "pairdist_" + Astr + "_particles_V0=" + V0string + ".pdf"
        )
    elif nV == 1:
        plot_filedm = (
            plot_folder + "pairdist_" + Astr + "_particles_s=" + s_string + ".pdf"
        )
    else:
        plot_filedm = (
            plot_folder
            + "pairdist_"
            + Astr
            + "_particles_V0="
            + V0string
            + "_s="
            + s_string
            + ".pdf"
        )

    if os.path.exists(plot_filedm):
        os.remove(plot_filedm)

    fig, ax = plt.subplots()
    fcont = ax.contourf(xx, xx, pair_dist, cmap="coolwarm")
    ax.contour(xx, xx, pair_dist, colors="k")
    ax.axis("square")
    fig.colorbar(fcont, ax=ax)
    ax.set_xlabel("Position, $x_1$ [ho units]")
    ax.set_ylabel("Position, $x_2$ [ho units]")
    ax.set_title(
        "Density matrix, A=" + Astr + ", $V_0=$" + V0string + ", $s=$" + str(s)
    )
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    plt.savefig(plot_filedm)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    A = 2
    V0 = -20
    s = 0.5
    for i in range(2, 7):
        for j in [0]:
            main(A=i, V0=j, s=s)
