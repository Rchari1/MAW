"""
Black-hole observables for SERPENS simulations.

Provides post-simulation analysis utilities for star--BH binary systems:
  * Bondi-Hoyle-Lyttleton capture cross-section diagnostics
  * Observer-frame line-of-sight projection at a specified binary inclination
  * Doppler line-profile synthesis from particle velocities
"""

import numpy as np


G = 6.6743e-11


def bondi_hoyle_radius(M_bh, v_rel):
    """
    Bondi-Hoyle-Lyttleton accretion radius for a point mass moving through a wind.

        r_BHL = 2 G M / v_rel^2

    Arguments
    ---------
    M_bh : float
        Accretor mass [kg].
    v_rel : float or array-like
        Relative velocity between accretor and wind [m/s].

    Returns
    -------
    r_bhl : same shape as v_rel
        Capture radius [m].
    """
    v_rel = np.asarray(v_rel)
    return 2.0 * G * M_bh / v_rel ** 2


def bhl_mass_capture_rate(M_bh, rho_wind, v_rel):
    """
    Analytical Bondi-Hoyle-Lyttleton mass capture rate.

        Mdot_BHL = pi r_BHL^2 rho_wind v_rel

    Arguments
    ---------
    M_bh : float
        Accretor mass [kg].
    rho_wind : float
        Local wind density [kg/m^3].
    v_rel : float
        Relative velocity [m/s].

    Returns
    -------
    Mdot : float
        Capture rate [kg/s].
    """
    r_bhl = bondi_hoyle_radius(M_bh, v_rel)
    return np.pi * r_bhl ** 2 * rho_wind * v_rel


def captured_particle_mask(particle_positions, particle_velocities,
                            bh_position, bh_velocity, M_bh, factor=1.0):
    """
    Flag test particles that lie within the Bondi-Hoyle-Lyttleton capture cross-section
    of the accretor.

    Arguments
    ---------
    particle_positions : (N, 3) array
        Particle positions [m].
    particle_velocities : (N, 3) array
        Particle velocities [m/s].
    bh_position : (3,) array
        BH position [m].
    bh_velocity : (3,) array
        BH velocity [m/s].
    M_bh : float
        BH mass [kg].
    factor : float
        Multiplier on r_BHL — useful to test more inclusive capture criteria
        (e.g. factor=2 doubles the effective capture radius).

    Returns
    -------
    mask : (N,) boolean array
        True if the particle is within the BHL capture radius for its current v_rel.
    """
    r_vec = np.asarray(particle_positions) - np.asarray(bh_position)
    v_vec = np.asarray(particle_velocities) - np.asarray(bh_velocity)
    r = np.linalg.norm(r_vec, axis=1)
    v_rel = np.linalg.norm(v_vec, axis=1)
    r_bhl = bondi_hoyle_radius(M_bh, v_rel)
    return r < factor * r_bhl


def project_to_observer(positions, velocities, inclination, omega=0.0, Omega=0.0):
    """
    Project simulation-frame Cartesian positions and velocities onto the plane of the sky
    for an observer.

    The simulation z-axis is the orbital angular-momentum axis. We rotate so that the
    observer's line of sight is at inclination `inclination` (in radians) from this axis.

    Arguments
    ---------
    positions : (N, 3) array
    velocities : (N, 3) array
    inclination : float
        Binary inclination [rad]. inclination=0 -> face-on, pi/2 -> edge-on.
    omega : float
        Argument of periapsis rotation [rad] (rotation in orbital plane).
    Omega : float
        Longitude of ascending node [rad] (rotation about line of sight).

    Returns
    -------
    sky_positions : (N, 3) array
        Components: (X_sky, Y_sky, Z_los) where Z_los is along the line of sight.
    los_velocities : (N,) array
        Velocity component along the observer line of sight [m/s], positive = redshift.
    """
    pos = np.asarray(positions)
    vel = np.asarray(velocities)

    # Rotate by argument of periapsis around z (orbital plane axis)
    co, so = np.cos(omega), np.sin(omega)
    R_omega = np.array([[co, -so, 0], [so, co, 0], [0, 0, 1]])

    # Inclination tilt: rotate around the new x-axis
    ci, si = np.cos(inclination), np.sin(inclination)
    R_i = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])

    # Longitude of ascending node: rotate around line of sight (final z-axis)
    cO, sO = np.cos(Omega), np.sin(Omega)
    R_Omega = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])

    R = R_Omega @ R_i @ R_omega

    sky_positions = pos @ R.T
    sky_velocities = vel @ R.T
    los_velocities = sky_velocities[:, 2]  # +z is towards observer
    return sky_positions, los_velocities


def doppler_line_profile(los_velocities, weights=None, rest_wavelength=None,
                          v_min=-1500e3, v_max=1500e3, n_bins=200,
                          thermal_broadening=0.0):
    """
    Build a Doppler line profile from particle line-of-sight velocities.

    Arguments
    ---------
    los_velocities : (N,) array
        Line-of-sight velocity per particle [m/s], positive = receding.
    weights : (N,) array or None
        Per-particle weight (e.g. superparticle mass).
    rest_wavelength : float or None
        If set [m or AA], output is also returned in wavelength units.
    v_min, v_max : float
        Profile velocity limits [m/s].
    n_bins : int
        Number of velocity bins.
    thermal_broadening : float
        Optional Gaussian sigma [m/s] for additional thermal broadening.

    Returns
    -------
    v_centers : (n_bins,) array
        Bin centers [m/s].
    flux : (n_bins,) array
        Profile (sum of weights per bin), optionally Gaussian-smoothed.
    wavelengths : (n_bins,) array or None
        Doppler-shifted wavelengths if rest_wavelength is set, else None.
    """
    v_los = np.asarray(los_velocities)
    if weights is None:
        weights = np.ones_like(v_los)
    else:
        weights = np.asarray(weights)

    edges = np.linspace(v_min, v_max, n_bins + 1)
    flux, _ = np.histogram(v_los, bins=edges, weights=weights)
    v_centers = 0.5 * (edges[:-1] + edges[1:])

    if thermal_broadening > 0:
        from scipy.ndimage import gaussian_filter1d
        sigma_bins = thermal_broadening / (edges[1] - edges[0])
        flux = gaussian_filter1d(flux, sigma_bins)

    wavelengths = None
    if rest_wavelength is not None:
        c = 2.998e8
        wavelengths = rest_wavelength * (1.0 + v_centers / c)

    return v_centers, flux, wavelengths
