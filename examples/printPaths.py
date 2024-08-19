import numpy as np


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
        "RightAngle": {
            "true": np.column_stack((np.where(t < 0.5, 2*t, 1), np.where(t >= 0.5, 2*(t-0.5), 0))),
            "noisy": add_anisotropic_noise(np.column_stack((np.where(t < 0.5, 2*t, 1), np.where(t >= 0.5, 2*(t-0.5), 0))), noise_sigma)
        },
        "FigureEight": {
            "true": np.column_stack((np.sin(2*np.pi*t), np.sin(4*np.pi*t)/2)),
            "noisy": add_anisotropic_noise(np.column_stack((np.sin(2*np.pi*t), np.sin(4*np.pi*t)/2)), noise_sigma)
        },
        "SmoothCurve": {
            "true": np.column_stack((np.cos(np.pi * t), np.sin(np.pi * t))),
            "noisy": add_anisotropic_noise(np.column_stack((np.cos(np.pi * t), np.sin(np.pi * t))), noise_sigma)
        },
        "StraightLine": {
            "true": np.column_stack((t, t)),
            "noisy": add_anisotropic_noise(np.column_stack((t, t)), noise_sigma)
        },
        "Spiral": {
            "true": np.column_stack((t * np.cos(2 * np.pi * t / np.log(t + 1)), t * np.sin(2 * np.pi * t / np.log(t + 1)))),
            "noisy": add_anisotropic_noise(np.column_stack((t * np.cos(2 * np.pi * t / np.log(t + 1)), t * np.sin(2 * np.pi * t / np.log(t + 1)))), noise_sigma)
        },
    }
    
    return shapes

def print_cpp_vector(shape_name, shape_data):
    print(f"const std::vector<PRS::DataTypes::Vector2D> {shape_name}NoisyPath = {{")
    for point in shape_data:
        print(f"    {{{point[0]:.6f}, {point[1]:.6f}}},")
    print("};")
    print()

# Generate shapes
n = 100  # number of points
noise_sigma = 0.05  # noise level
shapes = generate_shapes(n, noise_sigma)

# Print each noisy shape as a C++ vector
for shape_name, shape_data in shapes.items():
    print_cpp_vector(shape_name, shape_data["noisy"])