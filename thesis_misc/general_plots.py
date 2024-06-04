import matplotlib.pyplot as plt
import numpy as np

from nqs.state.utils import plot_style


# # Define the x values for the plot
# x = np.linspace(-3, 3, 400)


# # Define the Target distribution (e.g., a normal distribution)
# def target_distribution(x):
#     return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)


# def envelope_distribution(x):
#     return 1.4 * target_distribution(x + 0.5) + 1.3 * target_distribution(x - 2.5) / 1.2


# # mycolor rgb
mycolor = (0.5, 0.5, 0.74)

# # Plotting
# plt.plot(x, target_distribution(x), label="Target", color=mycolor)
# plt.plot(x, envelope_distribution(x), linestyle="--", label="Envelope", color="black")

# # Vertical lines for decision points
# plt.plot([-1, -1], [0.0, target_distribution(-1)], color="green", linestyle="--")
# plt.plot(
#     [-1, -1],
#     [target_distribution(-1), envelope_distribution(-1)],
#     color="red",
#     linestyle="--",
# )

# # Annotating the decision points
# plt.text(
#     -0.35,
#     0.42,
#     "Reject",
#     verticalalignment="bottom",
#     horizontalalignment="right",
#     color="red",
#     fontsize=12,
# )
# plt.text(
#     -0.35,
#     0.2,
#     "Accept",
#     verticalalignment="bottom",
#     horizontalalignment="right",
#     color="green",
#     fontsize=12,
# )

# # Adding labels and legend
# plt.xlabel("X values")
# plt.ylabel("Probability")
# # plt.title('Rejection Sampling Visualization')
# plt.legend()

# plot_style.save("rejection_sampling.pdf")
# # Show the plot
# plt.show()


#########
from scipy.fft import fft, fftshift, fftfreq
from scipy.interpolate import interp1d


# Parameters for the cat state
alpha = 2.0
phi = np.pi / 4
hbar = 1.0


# Define the wavefunction for the cat state
def psi_cat(q, alpha, phi):
    return (1 / np.sqrt(2)) * (
        np.exp(-((q - alpha) ** 2)) + np.exp(-((q + alpha) ** 2)) * np.exp(1j * phi)
    )


# Define the Wigner function using Fourier transform
def wigner_function(q, p, psi):
    x = q[:, None]
    y = q[None, :]
    psi_x = psi(x)
    psi_y = psi(y)
    phase = np.exp(-2j * np.outer(p, q) / hbar)
    return (1 / np.pi * hbar) * np.sum(psi_x * np.conj(psi_y) * phase, axis=1).real


# Create a grid for q and p
q = np.linspace(-5, 5, 1000)
p = np.linspace(-5, 5, 5000)
Q, P = np.meshgrid(q, p)

# Compute the Wigner function values
W = np.array([wigner_function(q, p_i, lambda x: psi_cat(x, alpha, phi)) for p_i in p])

# Compute the marginal probability densities
psi_q = np.abs(psi_cat(q, alpha, phi)) ** 2

# Use finer resolution for the Fourier transform to achieve smoother results
q_fine = np.linspace(-5, 5, 5000)
psi_q_fine = np.abs(psi_cat(q_fine, alpha, phi)) ** 2
psi_p_complex = fftshift(fft(psi_q_fine))
psi_p = np.abs(psi_p_complex) ** 2
p_normalized = fftshift(fftfreq(q_fine.size, d=(q_fine[1] - q_fine[0]))) * (2 * np.pi)

# Normalize psi_p to fit within the Wigner function's z-axis limits
max_W = np.max(W)
psi_q_scaled = psi_q * max_W / np.max(psi_q)
psi_p_scaled = psi_p * max_W / np.max(psi_p)

# Smooth psi_p by interpolation
psi_p_interp = interp1d(p_normalized, psi_p_scaled, kind="cubic")
p_interp = np.linspace(-5, 5, len(p_normalized) * 10)
psi_p_smooth = psi_p_interp(p_interp)

# Clip the smoothed psi_p values to stay within the maximum Wigner function value
psi_p_smooth = np.clip(psi_p_smooth, 0, max_W)

# Plot the Wigner function and the marginal densities
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 3D plot of the Wigner function
surf = ax.plot_surface(
    Q, P, W, cmap="BuPu", edgecolor="none", alpha=0.8, linewidth=0.4, color="grey"
)

# Set limits to ensure the projections are visible
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([0, max_W])

# Project position probability density on the y=5 plane
ax.plot(
    q,
    np.full_like(q, 5),
    psi_q_scaled,
    color=mycolor,
    lw=2,
    label=r"$|\langle q | \psi \rangle|^2$",
)

# Project smoothed momentum probability density on the x=-5 plane
p_x = np.full_like(p_interp, -5)
p_y = p_interp
p_z = psi_p_smooth
ax.plot(p_x, p_y, p_z, color=mycolor, lw=2, label=r"$|\langle p | \psi \rangle|^2$")


# Plot settings
ax.set_xlabel("q")
ax.set_ylabel("p")
ax.set_zlabel("W(q, p)")
# ax.legend()
# instead of adding legend, add text close to the curves
ax.text(5, 1, max_W, r"$|\langle q | \psi \rangle|^2$", fontsize=14)
ax.text(-5, -4.4, max_W, r"$|\langle p | \psi \rangle|^2$", fontsize=14)

# Add color bar for Wigner function

# remove walls
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# remove ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


# remove edges
ax.xaxis.set_tick_params(size=0)
ax.yaxis.set_tick_params(size=0)
ax.zaxis.set_tick_params(size=0)

# remove frames


# remove spines
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# set tight layout
plt.tight_layout()
plot_style.save("QM_section_cover.pdf")
# # Display the plot
# plt.show()

# # Keep the plot open
# plt.ioff()
# plt.show()
