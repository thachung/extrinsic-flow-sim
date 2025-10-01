import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
N = 2000  # Number of points on the curve
T = 2.0  # Total simulation time
dt = 0.002  # Time step
steps = int(T / dt)  # Number of time steps
plot_every = steps // 8  # Plot at intervals


# Initial curve: ellipse
a, b = 2.0, 1.0
theta = np.linspace(0, 2 * np.pi - 1e-10, N)
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
    L = diags(diagonals, offsets, shape=(N, N), format="csr")
    # Periodic boundary conditions
    L = L.tolil()
    L[0, -1] = 1
    L[-1, 0] = 1
    return L.tocsr()


# Precompute Laplacian
L = laplacian_matrix(N)

# For uniform parameter spacing
ds = 2 * np.pi / N

# For plotting
plt.figure(figsize=(8, 8))
plt.axis("equal")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title("Curve Shortening Flow (Semi-Implicit Scheme)")
plt.xlabel("x")
plt.ylabel("y")

# Plot initial curve
plt.plot(x, y, label="t=0")


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
    s_uniform = np.linspace(0, 1 - 1e-10, N)

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
for step in range(1, steps + 1):
    # Solve (I - dt/ds^2 * L) x_new = x, and similarly for y
    A = diags([1], [0], shape=(N, N), format="csr") - (dt / ds**2) * L
    x_new = spsolve(A, x)
    y_new = spsolve(A, y)
    # Reparametrize to maintain uniform spacing
    x, y, ds = reparametrize(x_new, y_new)

    # Plot at intervals
    if step % plot_every == 0 or step == steps:

        plt.plot(x, y, label=f"t={step*dt:.2f}")


plt.legend()
plt.show()
