import matplotlib.pyplot as plt
import numpy as np
from scipy.special import fresnel

# importing module
import sys
 
# appending a path
sys.path.append('ccma')
from ccma import CCMA

# Constant Seed
np.random.seed(42)

# Create a noisy 2d-path
n = 75
sigma = 0.01
# shapes = generate_shapes(n, sigma)

# # Now you can access each shape like this:
# right_angle_true = shapes["straight_changing_curvature_curve"]["true"]
# right_angle_noisy = shapes["straight_changing_curvature_curve"]["true"]
simplePath = np.array([[1.0, 1.0], [5.0, 5.0], [2.0, 1.0]])
# Create the CCMA-filter object
w_ma = 3
w_cc = 2
ccma = CCMA(w_ma, w_cc, distrib="hanning")


# Filter points with and w/o boundaries
ccma_points = ccma.filter(simplePath, mode="padding")
# ccma_points_wo_padding = ccma.filter(points_errors, mode="none")
# ma_points = ccma.filter(points_errors, cc_mode=False)

# Visualize results
# plt.plot(*right_angle_true.T, "r-o", linewidth=4, alpha=0.3, color='yellow', markersize=10, label="original TRUE")
plt.plot(*simplePath.T, "r-o", linewidth=3, alpha=0.3, color='red', markersize=10, label="original Noise")
plt.plot(*ccma_points.T, linewidth=6, alpha=1.0, color="orange", label=f"ccma-smoothed ({w_ma}, {w_cc})")
# plt.plot(*ccma_points_wo_padding.T, linewidth=3, alpha=0.5, color="b", label=f"ccma-smoothed ({w_ma}, {w_cc})")
# plt.plot(*ma_points.T, linewidth=2, alpha=0.5, color="green", label=f"ma-smoothed ({w_ma})")
# average_error = calculate_perpendicular_error(right_angle_true, ccma_points)

# print(f"Average error between true path and CCMA path: {average_error:.6f}")
# General settings
plt.grid(True)
plt.gca().set_aspect('equal')
plt.legend()
plt.tight_layout()
plt.gcf().set_size_inches(12, 6)
plt.xlabel("x")
plt.ylabel("y")
plt.title("CCMA - straight_changing_curvature_curve (2d)")
# print average error in the bottom right corner of the figure
# plt.text(0.95, 0.05, f"Average error: {average_error:.6f}")

plt.show()