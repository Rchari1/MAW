"""
Black-hole observables for SERPENS simulations.

Provides post-simulation analysis utilities for star--BH binary systems:
  * Bondi-Hoyle-Lyttleton capture cross-section diagnostics
  * Observer-frame line-of-sight projection at a specified binary inclination
  * Doppler line-profile synthesis from particle velocities
  * X-ray photoionization lifetime estimates for trace species
  * Physical-units column-density maps via superparticle weighting

References:
  * Edgar (2004) New Astron. Reviews 48, 843 — BHL accretion review
  * Verner et al. (1996) ApJ 465, 487 — atomic photoionization cross sections
  * Marsch (2006) Living Reviews Solar Physics 3, 1 — kinetic stellar winds
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


# ---------------------------------------------------------------------------
# X-ray photoionization lifetimes near an accreting compact object.
# Adopts an isotropic point-source approximation for the disk X-ray emission
# and integrates a power-law photon spectrum over an effective species cross
# section. Suitable for first-order lifetime estimates; for spectral-resolved
# work this should be replaced with a Verner+1996 cross-section table.
# ---------------------------------------------------------------------------

# Approximate effective X-ray photoionization cross sections [cm^2] at ~1 keV
# Order-of-magnitude values from Verner+1996 / Wilms+2000; sufficient for
# methods-paper lifetime scaling.
SPECIES_XRAY_SIGMA = {
    'H':   6.3e-22,
    'He':  7.4e-22,
    'C':   2.0e-19,
    'N':   3.7e-19,
    'O':   5.6e-19,
    'Fe':  3.5e-18,
    'Si':  9.0e-19,
}


def xray_photoionization_lifetime(L_X, distance_to_source, sigma_X=1e-19,
                                   mean_photon_energy_eV=1000.0):
    """
    Mean photoionization lifetime tau = 1 / (sigma_X * Phi_X) for a trace
    species at given distance from an isotropic X-ray point source.

    Arguments
    ---------
    L_X : float
        Source X-ray luminosity [erg/s].
    distance_to_source : float
        Distance from the source [m].
    sigma_X : float
        Effective photoionization cross section at <E_photon> [cm^2].
        Default 1e-19 (typical for low-Z metals at 1 keV).
    mean_photon_energy_eV : float
        Characteristic photon energy [eV]. Default 1000 (1 keV).

    Returns
    -------
    tau : float
        Mean lifetime [s]. Returns np.inf when L_X = 0.
    """
    if L_X <= 0:
        return np.inf
    erg_per_eV = 1.602e-12
    distance_cm = distance_to_source * 100.0
    mean_E = mean_photon_energy_eV * erg_per_eV
    photon_flux = L_X / (4.0 * np.pi * distance_cm ** 2 * mean_E)   # photons/cm^2/s
    rate = sigma_X * photon_flux
    return 1.0 / rate


def species_lifetime_in_disk_field(species_name, L_X, distance_to_source,
                                     mean_photon_energy_eV=1000.0):
    """
    Convenience wrapper using the SPECIES_XRAY_SIGMA table.
    """
    sigma = SPECIES_XRAY_SIGMA.get(species_name, 1.0e-19)
    return xray_photoionization_lifetime(L_X, distance_to_source,
                                          sigma_X=sigma,
                                          mean_photon_energy_eV=mean_photon_energy_eV)


# ---------------------------------------------------------------------------
# Physical-units column density maps.
# Uses the SERPENS superparticle weighting:
#     w_real_per_super = Mdot_total * t_sim / (n_per_spawn * n_spawns * m_species)
# We pass that weight per particle to convert (count per pixel) into (cm^-2).
# ---------------------------------------------------------------------------

def column_density_map(sky_xy, weights, extent_au, n_bins=80, smooth_sigma=1.0):
    """
    Build a 2D column density map in [particles / cm^2] from sky-projected
    particle positions and per-particle physical weights.

    Arguments
    ---------
    sky_xy : (N, 2) array
        Sky-plane positions [m].
    weights : (N,) array
        Real particles represented per superparticle.
    extent_au : float
        Half-width of the map in AU.
    n_bins : int
        Pixels per side.
    smooth_sigma : float
        Optional Gaussian smoothing sigma in pixels.

    Returns
    -------
    H : (n_bins, n_bins) array
        Column density [particles / cm^2] per pixel.
    edges_au : (n_bins+1,) array
        Pixel edges in AU.
    """
    AU_m = 1.496e11
    AU_cm = AU_m * 100.0
    half_m = extent_au * AU_m
    edges_m = np.linspace(-half_m, half_m, n_bins + 1)
    pixel_area_cm2 = ((edges_m[1] - edges_m[0]) * 100.0) ** 2

    H, _, _ = np.histogram2d(sky_xy[:, 0], sky_xy[:, 1],
                              bins=[edges_m, edges_m], weights=weights)
    H = H / pixel_area_cm2

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=smooth_sigma)

    edges_au = edges_m / AU_m
    return H, edges_au


# ---------------------------------------------------------------------------
# Composition helpers — solar abundances and FIP enhancement.
# Reference: Asplund, Grevesse, Sauval, Scott (2009), ARAA 47, 481.
# Low-FIP elements (FIP < 10 eV) like Fe, Mg, Si are observed to be enhanced
# by factor ~3-4 in the slow solar wind relative to photospheric abundances.
# ---------------------------------------------------------------------------

# Photospheric mass fractions (approximate, solar)
PHOTOSPHERIC_MASS_FRACTION = {
    'H':  0.7381,
    'He': 0.2485,
    'C':  2.4e-3,
    'N':  7.0e-4,
    'O':  5.7e-3,
    'Si': 7.1e-4,
    'Fe': 1.3e-3,
}

# First ionization potentials [eV]
FIP_eV = {
    'H': 13.60, 'He': 24.59, 'C': 11.26, 'N': 14.53,
    'O': 13.62, 'Si': 8.15,  'Fe': 7.90,
}


def fip_abundances(Mdot_total, fip_factor=3.5, threshold_eV=10.0,
                   species=None):
    """
    Apply FIP-effect enhancement to photospheric abundances and return per-species
    mass-loss rates.

    Low-FIP elements (FIP < threshold) get their mass fraction boosted by
    `fip_factor`; high-FIP elements (H, He, C, N, O at solar abundances) are
    unchanged. Resulting fractions are renormalized to preserve `Mdot_total`.

    Arguments
    ---------
    Mdot_total : float
        Total wind mass-loss rate [kg/s].
    fip_factor : float
        Enhancement factor for low-FIP elements (default 3.5).
    threshold_eV : float
        FIP below which an element is enhanced (default 10 eV).
    species : list[str] or None
        Subset of species to include. If None, returns all in PHOTOSPHERIC_MASS_FRACTION.

    Returns
    -------
    dict
        {species_name: Mdot_species [kg/s]}
    """
    if species is None:
        species = list(PHOTOSPHERIC_MASS_FRACTION.keys())

    fractions = {}
    for s in species:
        if s not in PHOTOSPHERIC_MASS_FRACTION:
            continue
        boost = fip_factor if FIP_eV.get(s, 99.0) < threshold_eV else 1.0
        fractions[s] = PHOTOSPHERIC_MASS_FRACTION[s] * boost

    norm = sum(fractions.values())
    return {s: Mdot_total * f / norm for s, f in fractions.items()}


# ---------------------------------------------------------------------------
# Predicted X-ray luminosity from BHL accretion rate.
# Includes radiative-efficiency suppression for low Mdot/Mdot_Edd (RIAF/ADAF
# regime, Narayan & Yi 1995; Yuan & Narayan 2014).
# ---------------------------------------------------------------------------

SIGMA_T = 6.6524e-29   # Thomson cross section [m^2]
M_PROTON = 1.673e-27   # [kg]


def eddington_mdot(M_bh, eta=0.1):
    """Eddington mass accretion rate [kg/s].

    Mdot_Edd = L_Edd / (eta c^2) with L_Edd = 4 pi G M m_p c / sigma_T.
    """
    G = 6.6743e-11
    c = 2.998e8
    L_edd = 4.0 * np.pi * G * M_bh * M_PROTON * c / SIGMA_T
    return L_edd / (eta * c ** 2)


def radiative_efficiency(Mdot, M_bh, eta_thin=0.1, transition_ratio=1e-2):
    """
    Radiative efficiency eta(Mdot) interpolating between the standard thin-disk
    value (eta_thin ~ 0.1) at high accretion rates and the RIAF/ADAF
    suppressed regime where eta ∝ Mdot / Mdot_Edd at low rates.

    Below `transition_ratio` x Mdot_Edd, eta declines linearly with Mdot:
        eta(m) = eta_thin * (m / transition_ratio)
    where m = Mdot / Mdot_Edd. For dormant BHs like Gaia BH1 this drops the
    predicted L_X by orders of magnitude relative to the thin-disk assumption.

    Reference: Narayan & Yi (1995); Yuan & Narayan (2014) ARAA.
    """
    Mdot_Edd = eddington_mdot(M_bh, eta=eta_thin)
    m = Mdot / Mdot_Edd
    if m >= transition_ratio:
        return eta_thin
    return eta_thin * (m / transition_ratio)


def xray_luminosity_from_mdot(Mdot, M_bh, eta_thin=0.1):
    """
    Predicted X-ray luminosity from a given accretion rate, accounting for
    ADAF suppression at low Mdot.

    Returns
    -------
    L_X : float [erg/s]
    """
    c = 2.998e8
    eta = radiative_efficiency(Mdot, M_bh, eta_thin=eta_thin)
    L_X_SI = eta * Mdot * c ** 2          # Joules/s = W
    return L_X_SI * 1.0e7                  # to erg/s


# ---------------------------------------------------------------------------
# Conservation diagnostics — for self-tests of GR + integrator behaviour.
# ---------------------------------------------------------------------------

def system_diagnostics(sim, com_reference=None):
    """
    Compute conservation diagnostics for the active (gravitating) bodies of a
    SERPENS / REBOUND simulation.

    Returns
    -------
    dict with keys:
      'energy'    : total mechanical energy [J]
      'L_total'   : magnitude of total angular momentum vector [kg m^2 / s]
      'L_vec'     : (3,) angular momentum vector
      'p_total'   : magnitude of total linear momentum [kg m/s]
      'com_drift' : displacement of center of mass from `com_reference` [m]
                     (or absolute COM position if com_reference is None)
    """
    n_active = sim.N_active if sim.N_active > 0 else sim.N
    G = 6.6743e-11

    masses = np.array([sim.particles[i].m for i in range(n_active)])
    pos = np.array([[sim.particles[i].x, sim.particles[i].y, sim.particles[i].z]
                    for i in range(n_active)])
    vel = np.array([[sim.particles[i].vx, sim.particles[i].vy, sim.particles[i].vz]
                    for i in range(n_active)])

    # Kinetic + potential
    KE = 0.5 * np.sum(masses[:, None] * vel ** 2)
    PE = 0.0
    for i in range(n_active):
        for j in range(i + 1, n_active):
            r = np.linalg.norm(pos[i] - pos[j])
            if r > 0:
                PE -= G * masses[i] * masses[j] / r
    energy = KE + PE

    L_vec = np.sum(masses[:, None] * np.cross(pos, vel), axis=0)
    L_total = np.linalg.norm(L_vec)

    p_vec = np.sum(masses[:, None] * vel, axis=0)
    p_total = np.linalg.norm(p_vec)

    M_total = masses.sum()
    com = (masses[:, None] * pos).sum(axis=0) / M_total
    if com_reference is not None:
        com_drift = np.linalg.norm(com - np.asarray(com_reference))
    else:
        com_drift = np.linalg.norm(com)

    return {
        'energy': energy,
        'L_total': L_total,
        'L_vec': L_vec,
        'p_total': p_total,
        'com_drift': com_drift,
    }
