import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
N = 2000  # Number of points
dt = 0.01  # Time step
T = 10  # Total simulation time
steps = int(T / dt)  # Number of time steps
L = 2 * np.pi  # Total length (assumed for uniform spacing)
plot_every = steps // 8  # Plot at intervals


# Create initial curve
t = np.linspace(0, 2 * np.pi - 1e-10, N)
R = 5.0
a = 3.0
x = R * np.cos(t)
y = R * np.sin(t)
z = a * np.sin(t)
curve = np.vstack((x, y, z)).T  # Shape (N, 3)


# Reparametrisation function
def reparametrize(x, y, z):
    # Compute cumulative arc length
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    dz = np.roll(z, -1) - z
    # calculate the norm by the Lorentz metric
    ds = np.sqrt(dx**2 + dy**2 - dz**2)
    s = np.cumsum(ds)
    s = np.insert(s, 0, 0)
    s /= s[-1]  # Normalize to [0,1]

    # Uniform parameter values
    s_uniform = np.linspace(0, 1 - 1e-10, N)

    # Interpolate to uniform spacing
    x_new = np.interp(s_uniform, s[:-1], x)
    y_new = np.interp(s_uniform, s[:-1], y)
    z_new = np.interp(s_uniform, s[:-1], z)

    # Compute new ds
    dx_new = np.roll(x_new, -1) - x_new
    dy_new = np.roll(y_new, -1) - y_new
    dz_new = np.roll(z_new, -1) - z_new
    ds = np.sqrt(dx_new**2 + dy_new**2 - dz_new**2)
    return x_new, y_new, z_new, np.mean(ds)


# For 3D plotting
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(projection="3d")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.axis("equal")


x, y, z, ds = reparametrize(curve[:, 0], curve[:, 1], curve[:, 2])
curve = np.vstack([x, y, z]).T
curve_new = copy.deepcopy(curve)
# Plot initial curve
ax1.plot(curve[:, 0], curve[:, 1], curve[:, 2], label=f"initial curve")

# Automatically determine axis limits from the data
x_limits = ax1.get_xlim()
y_limits = ax1.get_ylim()
z_limits = ax1.get_zlim()

# Find the global min and max to apply equal scaling
min_limit = min(x_limits[0], y_limits[0], z_limits[0])
max_limit = max(x_limits[1], y_limits[1], z_limits[1])

# Set all axes to the same limits
ax1.set_xlim(min_limit, max_limit)
ax1.set_ylim(min_limit, max_limit)
ax1.set_zlim(min_limit, max_limit)


# Precompute Laplacian matrix
main_diag = -2 * np.ones(N)
off_diag = np.ones(N - 1)
laplacian = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N))
laplacian = laplacian.tolil()
laplacian[0, -1] = 1
laplacian[-1, 0] = 1
laplacian = laplacian.tocsr()
laplacian_ds = laplacian / (ds**2)
A = (diags([1.0], [0], shape=(N, N)) - dt * laplacian_ds).tocsc()


# Time evolution using semi-implicit Euler
for step in range(1, steps + 1):
    for dim in range(3):
        curve_new[:, dim] = spsolve(A, curve[:, dim])
    x, y, z, ds = reparametrize(curve_new[:, 0], curve_new[:, 1], curve_new[:, 2])
    laplacian_ds = laplacian / (ds**2)
    A = (diags([1.0], [0], shape=(N, N)) - dt * laplacian_ds).tocsc()
    curve = np.vstack([x, y, z]).T
    # Plot at intervals
    if step % plot_every == 0 or step == steps:
        ax1.plot(curve[:, 0], curve[:, 1], curve[:, 2], label=f"t={step*dt:.3f}")

# Visualization
ax1.legend()
# Initial curve
plt.show()
