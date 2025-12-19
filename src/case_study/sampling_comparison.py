import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc


def sobol_sphere_volume_lowdisc(center, radius, n, seed=42):
    """Uniform volume sampling using 2D Sobol → disc."""
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    u = sampler.random(n=n)  # points in [0,1]^2

    # Transform to disc: r = R * sqrt(u1), theta = 2π * u2
    r = radius * np.sqrt(u[:, 0])  # sqrt for uniform area in 2D
    theta = 2 * np.pi * u[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    points = np.column_stack([x, y])
    return points + center


# ---------- Parameters ----------
center = np.array([0.0, 0.0])
radius = 5.0
D = 2
alpha = 1

# FDA original points (center + s1, s2 along first axis)
d = 0  # axis index (0‑based)
e_d = np.zeros(D)
e_d[d] = 1.0

shift = alpha * radius / np.sqrt(D) * np.ones(D)
s1 = center + shift
s2 = center - shift
original_points = np.array([center, s1, s2])

# FDA‑AS Sobol points (n = 2^ceil(log2(5*D)) = 16 for D=2)
n_sobol = 2 ** int(np.ceil(np.log2(5 * D)))
sobol_points = sobol_sphere_volume_lowdisc(center, radius, n=n_sobol, seed=1)

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

# Left: original FDA
circle1 = plt.Circle((center[0], center[1]), radius,
                     fill=False, color='black', linestyle='--')
ax1.add_patch(circle1)
ax1.scatter(original_points[:, 0], original_points[:, 1],
            c='blue', s=40, zorder=5, label='FDA points')
ax1.set_aspect('equal')
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.set_title('Original FDA sampling\n(center + $s_1$, $s_2$ a)')
#ax1.grid(True, alpha=0.3)
ax1.legend()

# Right: FDA‑AS Sobol
circle2 = plt.Circle((center[0], center[1]), radius,
                     fill=False, color='black', linestyle='--')
ax2.add_patch(circle2)
ax2.scatter(sobol_points[:, 0], sobol_points[:, 1],
            c='blue', s=40, zorder=5, label='Sobol points')
ax2.set_aspect('equal')
ax2.set_xlim(-6, 6)
ax2.set_ylim(-6, 6)
ax2.set_title(f'FDA‑AS Sobol sampling\n(n = {n_sobol})')
#ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig('sampling_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()