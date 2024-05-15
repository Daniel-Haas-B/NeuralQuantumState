import matplotlib.pyplot as plt
import numpy as np

from nqs.state.utils import plot_style


# Define the x values for the plot
x = np.linspace(-3, 3, 400)


# Define the Target distribution (e.g., a normal distribution)
def target_distribution(x):
    return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)


def envelope_distribution(x):
    return 1.4 * target_distribution(x + 0.5) + 1.3 * target_distribution(x - 2.5) / 1.2


# mycolor rgb
mycolor = (0.5, 0.5, 0.74)

# Plotting
plt.plot(x, target_distribution(x), label="Target", color=mycolor)
plt.plot(x, envelope_distribution(x), linestyle="--", label="Envelope", color="black")

# Vertical lines for decision points
plt.plot([-1, -1], [0.0, target_distribution(-1)], color="green", linestyle="--")
plt.plot(
    [-1, -1],
    [target_distribution(-1), envelope_distribution(-1)],
    color="red",
    linestyle="--",
)

# Annotating the decision points
plt.text(
    -0.35,
    0.42,
    "Reject",
    verticalalignment="bottom",
    horizontalalignment="right",
    color="red",
    fontsize=12,
)
plt.text(
    -0.35,
    0.2,
    "Accept",
    verticalalignment="bottom",
    horizontalalignment="right",
    color="green",
    fontsize=12,
)

# Adding labels and legend
plt.xlabel("X values")
plt.ylabel("Probability")
# plt.title('Rejection Sampling Visualization')
plt.legend()

plot_style.save("rejection_sampling.pdf")
# Show the plot
plt.show()
