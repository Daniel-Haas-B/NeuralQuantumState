import os

import numpy as np
import pandas as pd

DATA_PATH = "/Users/orpheus/Documents/Masters/NeuralQuantumState/data/fermion_polarized"


def get_groundstate(A: int, V0: int, datapath: str) -> float:
    r"""Method to get the analytical groundstate for the Harmonic Oscillator.
    It currently supports up to :math:`2 \leq A \leq 5` and :math:`-20 \leq V_{0} \leq +20`
    (in steps in 1) and :math:`\sigma_{0} = 0.5`

    :param A: The number of fermions
    :type A: int

    :param V0: The interaction strength
    :type V0: int

    :param datapath: The datapath of the groundstate energies file
    :type: str
    """
    if A < 2 or A > 5:
        # raise ValueError("Only have energies for 2 <= A <= 5")
        print("Only have energies for 2 <= A <= 5. Returning non-interacting value.")
        return A**2 / 2
    if V0 < -20 or V0 > 20:
        # raise ValueError("Only have energies for -20 <= V0 <= +20")
        print(
            "Only have energies for -20 <= V0 <= +20. Returning non-interacting value."
        )
        return A**2 / 2

    filestr = "%s%i%s" % (datapath, A, "p_20modes.txt")

    data = np.genfromtxt(filestr)

    idx = int(V0 + 20)

    gs = data[idx, 1]  # get gs values from .txt file
    return gs


V0 = 0  # args.V0
sigma0 = 0.5


for i in range(2, 6):
    for V0 in [-20, -10, 10, 20]:
        nfermions = i
        gs_CI = get_groundstate(A=i, V0=V0, datapath="groundstate/")

        data = {"A": nfermions, "V0": V0, "s": sigma0, "Energy": gs_CI}

        df = pd.DataFrame(data, index=[0])

        if not os.path.exists(DATA_PATH + "/data_ci.csv"):
            df.to_csv(DATA_PATH + "/data_ci.csv", mode="a", header=True)
        else:
            df.to_csv(DATA_PATH + "/data_ci.csv", mode="a", header=False)

        print(f"Groundstate Energy: {gs_CI}")
