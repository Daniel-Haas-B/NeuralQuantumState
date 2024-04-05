import h5py
import matplotlib.pyplot as plt

path = "/Users/haas/Documents/Masters/NQS/data/positions_VMC.h5"


with h5py.File(path, "r") as f:
    positions = f["positions"][:]

# scatter plot of the positions (its a 2D system)
positions = positions.reshape(-1, 2, 2)

plt.scatter(positions[:, 0, 0], positions[:, 0, 1], alpha=0.1)
plt.show()
