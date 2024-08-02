import matplotlib.pyplot as plt
import numpy as np

# importing module
import sys
 
# appending a path
sys.path.append('ccma')
from ccma import CCMA

# Constant Seed
np.random.seed(42)

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
    
    return np.mean(errors), np.max(errors)

# Create a noisy 2d-path
n = 75
sigma = 0.015
# After generating shapes and defining CCMA parameters
shapes = generate_shapes(n, sigma)
w_ma_values = [2, 3, 4, 6]
w_cc_values = [2, 3]

for shape_name, shape_data in shapes.items():
    true_path = shape_data["true"]
    noisy_path = shape_data["noisy"]
    
    print(f"\nProcessing shape: {shape_name}")
    
    # Calculate baseline error (without CCMA)
    baseline_error, baseline_maxXTrackerror= calculate_perpendicular_error(true_path, noisy_path)
    print(f"Baseline error (no CCMA): {baseline_error:.6f}")
    
    for w_ma in w_ma_values:
        for w_cc in w_cc_values:
            ccma = CCMA(w_ma, w_cc, distrib="hanning")
            ccma_points = ccma.filter(noisy_path, mode="none")
            
            average_error, maxXTrackerror  = calculate_perpendicular_error(true_path, ccma_points)
            error_improvement = (baseline_error - average_error) / baseline_error * 100
            maxTrackError_improvement = (baseline_maxXTrackerror - maxXTrackerror) / baseline_maxXTrackerror * 100
            
            print(f"CCMA (w_ma={w_ma}, w_cc={w_cc}) - Error: {average_error:.6f}, Improvement: {error_improvement:.2f}%, MaxXTrackError Improvement: {maxTrackError_improvement:.6f}%")