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

def generate_custom_shape1(n, profile):
    # Calculate the number of points for each segment
    n1 = n // 3  # Straight line
    n2 = n // 3  # Constant curve
    n3 = n - (n1 + n2)  # Perpendicular straight line (remaining points)

    # Generate the path segments
    t1 = np.linspace(0, 1, n1)
    x1 = t1 * 3
    y1 = np.zeros_like(x1)

    theta = np.linspace(0, np.pi/2, n2)
    r = 1
    x2 = x1[-1] + r * np.sin(theta)
    y2 = y1[-1] - r * (1 - np.cos(theta))

    x3 = np.ones(n3) * x2[-1]
    y3 = np.linspace(y2[-1], y2[-1] - 1, n3)

    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])

    path = np.column_stack((x, y))
    
    if profile == 1:
        # Evenly spaced points
        return path
    if profile == 2:
        # Simulate vehicle slowing down on curve
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumulative_distances = np.cumsum(distances)
        total_distance = cumulative_distances[-1]
        
        # Create a speed profile
        t = np.linspace(0, 1, n)
        curve_start = n1 / n
        curve_end = (n1 + n2) / n
        speed = 1 - 0.7 * np.exp(-((t - (curve_start + curve_end)/2) / ((curve_end - curve_start)/4))**2)
        
        # Generate new points based on speed profile
        new_t = np.cumsum(1 / speed)
        new_t = new_t / new_t[-1] * total_distance
        
        new_x = np.interp(new_t, cumulative_distances, x[1:])
        new_y = np.interp(new_t, cumulative_distances, y[1:])
        
        return np.column_stack((new_x, new_y))
    elif profile == 3:
        # Constant separation with perpendicular error
        tangent = np.gradient(path, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]
        perpendicular_noise = np.random.normal(0, 0.05, (len(path), 1)) * normal
        return path + perpendicular_noise

def generate_custom_shape2(n):
    # Calculate the number of points for each segment
    n1 = n // 3  # Straight line
    n2 = n // 3  # Changing curvature curve
    n3 = n - (n1 + n2)  # Perpendicular straight line (remaining points)

    # Generate the path segments
    t1 = np.linspace(0, 1, n1)
    x1 = t1 * 3
    y1 = np.zeros_like(x1)
    
    # Changing curvature curve (spiral-like)
    t = np.linspace(0, 1, n2)
    a = 2  # Controls the tightness of the spiral
    x2 = x1[-1] + t + (t**a) * np.sin(np.pi * t)
    y2 = y1[-1] - (t**a) * (1 - np.cos(np.pi * t))
    
    # Perpendicular straight line
    x3 = np.ones(n3) * x2[-1]
    y3 = np.linspace(y2[-1], y2[-1] - 1, n3)
    
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    
    return np.column_stack((x, y))


def generate_shapes(n, noise_sigma):
    t = np.linspace(0, 1, n)
    
    def add_anisotropic_noise(path, sigma):
        tangent = np.gradient(path, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]
        perpendicular_noise = np.random.normal(0, sigma, (n, 1)) * normal
        along_path_noise = np.random.normal(0, sigma * 0.1, (n, 2))  # Reduced noise along the path
        return path + perpendicular_noise + along_path_noise

    shapes = {
        "right_angle": {
            "true": np.column_stack((np.where(t < 0.5, 2*t, 1), np.where(t >= 0.5, 2*(t-0.5), 0))),
            "noisy": add_anisotropic_noise(np.column_stack((np.where(t < 0.5, 2*t, 1), np.where(t >= 0.5, 2*(t-0.5), 0))), noise_sigma)
        },
        "figure_eight": {
            "true": np.column_stack((np.sin(2*np.pi*t), np.sin(4*np.pi*t)/2)),
            "noisy": add_anisotropic_noise(np.column_stack((np.sin(2*np.pi*t), np.sin(4*np.pi*t)/2)), noise_sigma)
        },
        "smooth_curve": {
            "true": np.column_stack((np.cos(np.pi * t), np.sin(np.pi * t))),
            "noisy": add_anisotropic_noise(np.column_stack((np.cos(np.pi * t), np.sin(np.pi * t))), noise_sigma)
        },
        "straight_line": {
            "true": np.column_stack((t, t)),
            "noisy": add_anisotropic_noise(np.column_stack((t, t)), noise_sigma)
        },
        "spiral": {
            "true": np.column_stack((t * np.cos(2 * np.pi * t / np.log(t + 1)), t * np.sin(2 * np.pi * t / np.log(t + 1)))),
            "noisy": add_anisotropic_noise(np.column_stack((t * np.cos(2 * np.pi * t / np.log(t + 1)), t * np.sin(2 * np.pi * t / np.log(t + 1)))), noise_sigma)
        },
    }

    shapes.update({
        "straight_const_curve": {
            "true": generate_custom_shape1(n, 1),
            "noisy": add_anisotropic_noise(generate_custom_shape1(n, 1), noise_sigma)
        },
        "straight_changing_points_curve": {
            "true": generate_custom_shape1(n, 2),
            "noisy": add_anisotropic_noise(generate_custom_shape1(n, 2), noise_sigma)
        },
        "straight_const_curve_lat_noise": {
            "true": generate_custom_shape1(n, 3),
            "noisy": generate_custom_shape1(n, 3)  # Already has perpendicular noise
        },
        "straight_changing_curvature_curve": {
            "true": generate_custom_shape2(n),
            "noisy": add_anisotropic_noise(generate_custom_shape2(n), noise_sigma)
        }
    })
    
    return shapes

def calculate_perpendicular_error(true_path, predicted_path):
    # Find the closest point on the true path to the first predicted point
    start_idx = np.argmin(np.linalg.norm(true_path[:len(true_path)//2] - predicted_path[0], axis=1))
    
    errors = []
    for i in range(start_idx, start_idx + len(predicted_path) - 1):
        if i - start_idx >= len(predicted_path):
            break
        
        # Get two consecutive points on the true path
        if (len(true_path) - 1 > i) and np.all(~np.isnan(true_path[i:i+2])) and np.all(~np.isnan(predicted_path[i - start_idx:i+2])):
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
n = 75
sigma = 0.01
shapes = generate_shapes(n, sigma)

# Now you can access each shape like this:
right_angle_true = shapes["straight_changing_curvature_curve"]["true"]
right_angle_noisy = shapes["straight_changing_curvature_curve"]["true"]

# Create the CCMA-filter object
w_ma = 6
w_cc = 3
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
plt.title("CCMA - straight_changing_curvature_curve (2d)")
# print average error in the bottom right corner of the figure
plt.text(0.95, 0.05, f"Average error: {average_error:.6f}")

plt.show()