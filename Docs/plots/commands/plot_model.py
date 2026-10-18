
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

radius = np.linspace(0.01, 10, 1000)

# Plummer density, normalized by its central value
density = (1 + radius**2) ** (-2.5)

# Fraction of cluster mass enclosed within radius r
mass_fraction = radius**3 / (1 + radius**2) ** 1.5

# Figure 1: Plummer density profile
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(radius, density, linewidth=2)
ax.set_xlabel(r"$r/a$")
ax.set_ylabel(r"$\rho(r)/\rho(0)$")
ax.set_title("Plummer density profile")
ax.set_xlim(0, 10)
ax.set_ylim(bottom=0)
ax.grid(True, linestyle="--", alpha=0.35)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "plummer_density.pdf")
plt.close(fig)

# Figure 2: Enclosed mass fraction
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(radius, mass_fraction, linewidth=2)
ax.axhline(0.5, linestyle=":", linewidth=1.5, label="50% of total mass")
ax.set_xlabel(r"$r/a$")
ax.set_ylabel(r"$M(<r)/M$")
ax.set_title("Enclosed mass fraction")
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "plummer_mass_fraction.pdf")
plt.close(fig)

# Galactic Plummer potential and radial acceleration
x = radius
potential = -1 / np.sqrt(1 + x**2)
radial_acceleration = -x / (1 + x**2) ** 1.5

# Figure 3: Galactic potential
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, potential, linewidth=2)
ax.set_xlabel(r"$R/b$")
ax.set_ylabel(r"$\Phi(R)/(GM_{\rm gal}/b)$")
ax.set_title("Galactic Plummer potential")
ax.set_xlim(0, 10)
ax.grid(True, linestyle="--", alpha=0.35)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "galactic_potential.pdf")
plt.close(fig)

# Figure 4: Radial gravitational acceleration
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, radial_acceleration, linewidth=2)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel(r"$R/b$")
ax.set_ylabel(r"$a_R/(GM_{\rm gal}/b^2)$")
ax.set_title("Galactic radial acceleration")
ax.set_xlim(0, 10)
ax.grid(True, linestyle="--", alpha=0.35)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "galactic_acceleration.pdf")
plt.close(fig)

print(f"Plots saved in {OUTPUT_DIR}")
