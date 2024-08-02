import matplotlib.pyplot as plt
import numpy as np

# importing module
import sys
 
# appending a path
sys.path.append('ccma')
from ccma import CCMA

def generate_shapes(n, noise_sigma):
    t = np.linspace(0, 1, n)
    
    shapes = {
        "right_angle": {
            "true": np.column_stack((np.where(t < 0.5, 2*t, 1), np.where(t >= 0.5, 2*(t-0.5), 0))),
            "noisy": np.column_stack((np.where(t < 0.5, 2*t, 1), np.where(t >= 0.5, 2*(t-0.5), 0))) + np.random.normal(0, noise_sigma, (n, 2))
        },
        "figure_eight": {
            "true": np.column_stack((np.sin(2*np.pi*t), np.sin(4*np.pi*t)/2)),
            "noisy": np.column_stack((np.sin(2*np.pi*t), np.sin(4*np.pi*t)/2)) + np.random.normal(0, noise_sigma, (n, 2))
        },
        "smooth_curve": {
            "true": np.column_stack((np.cos(np.pi * t), np.sin(np.pi * t))),
            "noisy": np.column_stack((np.cos(np.pi * t), np.sin(np.pi * t))) + np.random.normal(0, noise_sigma, (n, 2))
        },
        "straight_line": {
            "true": np.column_stack((t, t)),
            "noisy": np.column_stack((t, t)) + np.random.normal(0, noise_sigma, (n, 2))
        },
        "spiral": {
            "true": np.column_stack((t * np.cos(2 * np.pi * t / np.log(t + 1)), t * np.sin(2 * np.pi * t / np.log(t + 1)))),
            "noisy": np.column_stack((t * np.cos(2 * np.pi * t / np.log(t + 1)), t * np.sin(2 * np.pi * t / np.log(t + 1)))) + np.random.normal(0, noise_sigma, (n, 2))
        },
    }
    
    return shapes

def calculate_perpendicular_error(true_path, predicted_path):
    # Find the closest point on the true path to the first predicted point
    start_idx = np.argmin(np.linalg.norm(true_path - predicted_path[0], axis=1))
    
    errors = []
    for i in range(start_idx, start_idx + len(predicted_path) - 1):
        if i - start_idx >= len(predicted_path):
            break
        
        # Get two consecutive points on the true path
        p1, p2 = true_path[i], true_path[i+1]
        
        # Calculate the vector of the true path segment
        v = p2 - p1
        
        # Normalize the vector
        v_norm = v / np.linalg.norm(v)
        
        # Calculate the perpendicular vector
        v_perp = np.array([-v_norm[1], v_norm[0]])
        
        # Calculate the perpendicular distance
        d = np.abs(np.dot(predicted_path[i - start_idx] - p1, v_perp))
        
        errors.append(d)
    
    return np.mean(errors)

# Create a noisy 2d-path
n = 100
sigma = 0.02
shapes = generate_shapes(n, sigma)

# Now you can access each shape like this:
right_angle_true = shapes["right_angle"]["true"]
right_angle_noisy = shapes["right_angle"]["noisy"]

# Create the CCMA-filter object
w_ma = 4
w_cc = 2
ccma = CCMA(w_ma, w_cc, distrib="hanning")


# Filter points with and w/o boundaries
ccma_points = ccma.filter(right_angle_noisy, mode="none")
# ccma_points_wo_padding = ccma.filter(points_errors, mode="none")
# ma_points = ccma.filter(points_errors, cc_mode=False)

# Visualize results
plt.plot(*right_angle_true.T, "r-o", linewidth=4, alpha=0.3, color='yellow', markersize=10, label="original TRUE")
plt.plot(*right_angle_noisy.T, "r-o", linewidth=3, alpha=0.3, color='red', markersize=10, label="original Noise")
plt.plot(*ccma_points.T, linewidth=6, alpha=1.0, color="orange", label=f"ccma-smoothed ({w_ma}, {w_cc})")
# plt.plot(*ccma_points_wo_padding.T, linewidth=3, alpha=0.5, color="b", label=f"ccma-smoothed ({w_ma}, {w_cc})")
# plt.plot(*ma_points.T, linewidth=2, alpha=0.5, color="green", label=f"ma-smoothed ({w_ma})")
average_error = calculate_perpendicular_error(right_angle_true, ccma_points)

print(f"Average error between true path and CCMA path: {average_error:.6f}")
# General settings
plt.grid(True)
plt.gca().set_aspect('equal')
plt.legend()
plt.tight_layout()
plt.gcf().set_size_inches(12, 6)
plt.xlabel("x")
plt.ylabel("y")
plt.title("CCMA - Example 1 (2d)")

plt.show()
