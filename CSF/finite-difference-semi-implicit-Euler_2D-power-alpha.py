import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

# Parameters
N = 2000  # Number of points on the curve
T = 0.5  # Total simulation time
dt = 0.01  # Time step
steps = int(T / dt)  # Number of time steps
plot_every = steps // 10  # Plot at intervals
alpha = 0.3333  # Power of curvature in the flow


# Initial curve: ellipse
a, b = 2, 1
theta = np.linspace(0 + 1e-10, 2 * np.pi - 1e-10, N, endpoint=False)
x = a * np.cos(theta)
y = b * np.sin(theta)


# Helper: build periodic Laplacian matrix
def laplacian_matrix(N):
    diagonals = [
        np.ones(N),  # main diagonal
        -2 * np.ones(N),  # center
        np.ones(N),  # off-diagonal
    ]
    offsets = [-1, 0, 1]
    L = diags_array(diagonals, offsets=offsets, shape=(N, N), format="csr")
    # Periodic boundary conditions
    L = L.tolil()
    L[0, -1] = 1
    L[-1, 0] = 1
    return L.tocsr()


# Precompute Laplacian
L = laplacian_matrix(N)

# For plotting
plt.figure(figsize=(8, 8))
plt.axis("equal")
plt.title(f"Curve Shortening Flow (Semi-Implicit Scheme) with power {alpha}")
plt.xlabel("x")
plt.ylabel("y")

# Plot initial curve
plt.plot(x, y, label="t=0")

# Automatic axis scaling
x_limits = plt.xlim()
y_limits = plt.ylim()

# Find the global min and max to apply equal scaling
min_limit = min(x_limits[0], y_limits[0])
max_limit = max(x_limits[1], y_limits[1])

# Set all axes to the same limits
plt.xlim(min_limit, max_limit)


# Reparametrisation function
def reparametrize(x, y):
    # Compute cumulative arc length
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)
    s = np.insert(s, 0, 0)
    s /= s[-1]  # Normalize to [0,1]

    # Uniform parameter values
    s_uniform = np.linspace(0 + 1e-10, 1 - 1e-10, N, endpoint=False)

    # Interpolate to uniform spacing
    x_new = np.interp(s_uniform, s[:-1], x)
    y_new = np.interp(s_uniform, s[:-1], y)

    # Compute new ds
    dx_new = np.roll(x_new, -1) - x_new
    dy_new = np.roll(y_new, -1) - y_new
    ds = np.sqrt(dx_new**2 + dy_new**2)
    return x_new, y_new, np.mean(ds)


x, y, ds = reparametrize(x, y)

# Time evolution
for step in tqdm(range(1, steps + 1)):
    L_ds = (1 / ds**2) * L
    x = x * 100
    y = y * 100
    k = np.sqrt((L_ds @ x) ** 2 + (L_ds @ y) ** 2) ** (
        alpha - 1
    )  # Curvature magnitude to the power alpha-1
    A = diags_array([1], offsets=[0], shape=(N, N), format="csr") - (dt / ds**2) * (
        diags_array([k], offsets=[0], shape=(N, N), format="csr") @ L
    )
    x_new = spsolve(A, x)
    y_new = spsolve(A, y)
    # Reparametrize to maintain uniform spacing
    x, y, ds = reparametrize(x_new, y_new)
    x = x / (100**alpha)
    y = y / (100**alpha)
    ds = ds / (100**alpha)
    # Plot at intervals
    if step % plot_every == 0 or step == steps:
        plt.plot(x, y, label=f"t={step*dt:.2f}")


plt.legend()
plt.show()
