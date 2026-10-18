#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rebound


# Project paths
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_PATH = SCRIPT_PATH.parents[3]
OUTPUT_PATH = SCRIPT_PATH.parents[1] / "orbital_methods_comparison.pdf"


G = 1
time_step = 0.001
integrator = "leapfrog"

galaxy_mass = 2000
galaxy_scale = 10

# Select orbital parameters
semi_major_axis = 60
eccentricity = 0.7


def galactic_potential(radius):
    """Gravitational potential of the galactic Plummer model."""
    return -G * galaxy_mass / np.sqrt(radius**2 + galaxy_scale**2)


def get_plummer_apocenter_velocity(r_apo, r_peri):
    """Calculate the apocenter velocity for the Plummer potential."""
    if np.isclose(r_apo, r_peri):
        return np.sqrt(
            G * galaxy_mass * r_apo**2
            / (r_apo**2 + galaxy_scale**2) ** 1.5
        )

    phi_apo = galactic_potential(r_apo)
    phi_peri = galactic_potential(r_peri)

    numerator = 2.0 * (phi_apo - phi_peri)
    denominator = 1.0 / r_peri**2 - 1.0 / r_apo**2

    angular_momentum_squared = numerator / denominator

    if angular_momentum_squared <= 0.0:
        raise ValueError(
            "Orbital parameters are not physically admissible."
        )

    return np.sqrt(angular_momentum_squared) / r_apo


def get_keplerian_apocenter_velocity(a, e):
    """Calculate the apocenter velocity for a Keplerian orbit."""
    return np.sqrt(
        G * galaxy_mass / a * (1.0 - e) / (1.0 + e)
    )


def galactic_forces(simulation_pointer):
    """Apply the galactic Plummer acceleration to each particle."""
    simulation = simulation_pointer.contents

    for index in range(simulation.N):
        particle = simulation.particles[index]

        radius_squared = (
            particle.x**2 + particle.y**2 + particle.z**2
        )
        denominator = (
            radius_squared + galaxy_scale**2
        ) ** 1.5

        particle.ax += (
            -G * galaxy_mass * particle.x / denominator
        )
        particle.ay += (
            -G * galaxy_mass * particle.y / denominator
        )
        particle.az += (
            -G * galaxy_mass * particle.z / denominator
        )


def integrate_orbits_with_safety_stop(v_plummer, v_kepler, simulation_duration):
    """
    Integrate both orbits simultaneously and abort early if their 
    positional difference exceeds a scientifically sensible threshold 
    (50% of the semi-major axis).
    """
    simulation = rebound.Simulation()
    simulation.G = G
    simulation.dt = time_step
    simulation.integrator = integrator
    simulation.gravity = "none"
    simulation.additional_forces = galactic_forces
    simulation.force_is_velocity_dependent = 0

    # Particle 0: Plummer-consistent velocity
    simulation.add(
        m=0.0,
        x=r_apo,
        y=0.0,
        z=0.0,
        vx=0.0,
        vy=v_plummer,
        vz=0.0,
    )
    # Particle 1: Keplerian velocity (point-mass approximation)
    simulation.add(
        m=0.0,
        x=r_apo,
        y=0.0,
        z=0.0,
        vx=0.0,
        vy=v_kepler,
        vz=0.0,
    )

    number_of_steps = int(round(simulation_duration / time_step))
    times = np.arange(number_of_steps + 1) * time_step

    positions_plummer = []
    radii_plummer = []
    positions_kepler = []
    radii_kepler = []
    recorded_times = []

    # Scientific divergence threshold: 50% of the semi-major axis 'a'
    divergence_threshold = 0.50 * semi_major_axis

    stopped_early = False
    stop_reason = ""

    p0 = simulation.particles[0]
    p1 = simulation.particles[1]

    recorded_times.append(times[0])
    positions_plummer.append([p0.x, p0.y])
    radii_plummer.append(np.hypot(p0.x, p0.y))
    positions_kepler.append([p1.x, p1.y])
    radii_kepler.append(np.hypot(p1.x, p1.y))

    for step in range(1, number_of_steps + 1):
        simulation.integrate(times[step])

        p0 = simulation.particles[0]
        p1 = simulation.particles[1]

        # Euclidean distance between the two trajectories (error metric)
        spatial_difference = np.sqrt(
            (p0.x - p1.x)**2 + (p0.y - p1.y)**2 + (p0.z - p1.z)**2
        )

        if spatial_difference > divergence_threshold:
            stopped_early = True
            stop_reason = (
                f"SAFETY ABORT: Trajectories diverged by > {divergence_threshold:.2f} "
                f"(50% of semi-major axis a={semi_major_axis}) "
                f"at t = {times[step]:.3f} (step {step})."
            )
            break

        recorded_times.append(times[step])
        positions_plummer.append([p0.x, p0.y])
        radii_plummer.append(np.hypot(p0.x, p0.y))
        positions_kepler.append([p1.x, p1.y])
        radii_kepler.append(np.hypot(p1.x, p1.y))

    return (
        np.array(recorded_times),
        np.array(positions_plummer),
        np.array(radii_plummer),
        np.array(positions_kepler),
        np.array(radii_kepler),
        stopped_early,
        stop_reason,
    )


# Validate parameters
if semi_major_axis <= 0.0 or not 0.0 <= eccentricity < 1.0:
    raise ValueError("Require a > 0 and 0 <= e < 1.")

if G <= 0.0 or galaxy_mass <= 0.0 or galaxy_scale <= 0.0:
    raise ValueError("G, galaxy mass and scale radius must be positive.")

if time_step <= 0.0:
    raise ValueError("The simulation time step must be positive.")

# Target apsidal distances
r_apo = semi_major_axis * (1.0 + eccentricity)
r_peri = semi_major_axis * (1.0 - eccentricity)

# Initial velocities
v_plummer = get_plummer_apocenter_velocity(r_apo, r_peri)
v_kepler = get_keplerian_apocenter_velocity(
    semi_major_axis, eccentricity
)

# Use the Keplerian period to set the plotting interval
reference_period = (
    2.0 * np.pi
    * np.sqrt(semi_major_axis**3 / (G * galaxy_mass))
)
simulation_duration = 2.0 * reference_period

# Integrate both orbits simultaneously with safety stop condition
(
    times,
    positions_plummer,
    radii_plummer,
    positions_kepler,
    radii_kepler,
    stopped_early,
    stop_reason,
) = integrate_orbits_with_safety_stop(v_plummer, v_kepler, simulation_duration)

# Print numerical results and safety status
print("Simulation parameters:")
print(f"  G: {G}")
print(f"  Galaxy mass: {galaxy_mass}")
print(f"  Galaxy scale radius: {galaxy_scale}")
print(f"  Integrator: {integrator}")
print(f"  Time step: {time_step}")
print()
print("Orbital parameters:")
print(f"  Semi-major-axis input: {semi_major_axis}")
print(f"  Eccentricity input: {eccentricity}")
print(f"  Target apocenter: {r_apo:.6f}")
print(f"  Target pericenter: {r_peri:.6f}")
print(f"  Plummer-consistent velocity: {v_plummer:.8f}")
print(f"  Keplerian velocity: {v_kepler:.8f}")
print()
if stopped_early:
    print(f"Status: {stop_reason}")
else:
    print("Status: Integration completed full duration without exceeding divergence threshold.")
print()

# Plot styling
plummer_color = "tab:blue"
kepler_color = "tab:orange"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Galactocentric distance over time
axes[0].plot(
    times,
    radii_plummer,
    color=plummer_color,
    label="Plummer-consistent velocity",
)
axes[0].plot(
    times,
    radii_kepler,
    color=kepler_color,
    label="Keplerian velocity",
)
axes[0].axhline(
    r_peri,
    color="black",
    linestyle="--",
    label="Target pericenter",
)
axes[0].axhline(
    r_apo,
    color="gray",
    linestyle=":",
    label="Initial apocenter",
)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Galactocentric distance")
axes[0].set_title("Orbital radius over time")
axes[0].grid(True, alpha=0.3)

# Orbital trajectories
axes[1].plot(
    positions_plummer[:, 0],
    positions_plummer[:, 1],
    color=plummer_color,
    label="Plummer-consistent velocity",
)
axes[1].plot(
    positions_kepler[:, 0],
    positions_kepler[:, 1],
    color=kepler_color,
    label="Keplerian velocity",
)
axes[1].scatter(
    [r_apo],
    [0.0],
    marker="*",
    s=130,
    color="red",
    edgecolors="white",
    linewidths=0.8,
    zorder=5,
    label="Common initial position",
)
axes[1].scatter(
    [0.0],
    [0.0],
    marker="*",
    s=130,
    color="green",
    edgecolors="white",
    linewidths=0.8,
    zorder=5,
    label="Galactic center",
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Orbital trajectories")
axes[1].set_aspect("equal", adjustable="datalim")
axes[1].grid(True, alpha=0.3)

# Shared legend
handles = []
labels = []

for axis in axes:
    axis_handles, axis_labels = axis.get_legend_handles_labels()

    for handle, label in zip(axis_handles, axis_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.085),
)

fig.tight_layout(rect=[0, 0.18, 1, 1])
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close(fig)

print(f"Plot saved to: {OUTPUT_PATH}")
