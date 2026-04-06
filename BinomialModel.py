import os
import numpy as np
from numpy import sqrt, sin, arcsin, cos, arccos, exp, pi, linspace, ceil
from plyfile import PlyData, PlyElement

def intersectBBox(ox, oy, oz, dx, dy, dz, sizex, sizey, sizez):

    # Intersection code below is adapted from Suffern (2007) Listing 19.1
    x0 = -0.5*sizex
    x1 = 0.5*sizex
    y0 = -0.5 * sizey
    y1 = 0.5 * sizey
    z0 = -1e-6
    z1 = sizez
    if dx == 0:
        a = 1e6
    else:
        a = 1.0 / dx
    if a >= 0:
        tx_min = (x0 - ox) * a
        tx_max = (x1 - ox) * a
    else:
        tx_min = (x1 - ox) * a
        tx_max = (x0 - ox) * a
    if dy == 0:
        b = 1e6
    else:
        b = 1.0 / dy
    if b >= 0:
        ty_min = (y0 - oy) * b
        ty_max = (y1 - oy) * b
    else:
        ty_min = (y1 - oy) * b
        ty_max = (y0 - oy) * b
    if dz == 0:
        c = 1e6
    else:
        c = 1.0 / dz
    if c >= 0:
        tz_min = (z0 - oz) * c
        tz_max = (z1 - oz) * c
    else:
        tz_min = (z1 - oz) * c
        tz_max = (z0 - oz) * c

    # find largest entering t value
    if tx_min > ty_min:
        t0 = tx_min
    else:
        t0 = ty_min
    if tz_min > t0:
        t0 = tz_min

    # find smallest exiting t value
    if tx_max < ty_max:
        t1 = tx_max
    else:
        t1 = ty_max
    if tz_max < t1:
        t1 = tz_max
    if t0 < t1 and t1 > 1e-6:
        if t0 > 1e-6:
            dr = t1-t0
        else:
            dr = t1
    else:
        dr = 0
    xe = ox + t1 * dx
    ye = oy + t1 * dy
    ze = oz + t1 * dz

    if dr == 0:
         raise Exception('Shouldnt be here')
    return dr, xe, ye, ze

def pathlengths(shape, scale_x, scale_y, scale_z, ray_zenith, ray_azimuth, nrays, outputfile=''):

    kEpsilon = 1e-5
    N = int(ceil(sqrt(nrays)))
    # Ray direction Cartesian unit vector
    dx = sin(ray_zenith) * cos(ray_azimuth)
    dy = sin(ray_zenith) * sin(ray_azimuth)
    dz = cos(ray_zenith)
    path_length = np.zeros(N*N)

    # Define bounding box depending on shape
    bbox_sizex = scale_x * (1.0 + kEpsilon)
    bbox_sizey = scale_y * (1.0 + kEpsilon)
    z_min = 0
    z_max = scale_z * (1.0 + kEpsilon)
    sx = bbox_sizex / N
    sy = bbox_sizey / N

    # loop over all rays, which originate at the bottom of the box
    for j in range(0, N):
        for i in range(0, N):
            # ray origin point
            ox = -0.5*bbox_sizex + (i+0.5)*sx
            oy = -0.5*bbox_sizey + (j+0.5)*sy
            oz = z_min - kEpsilon
            ze = 0
            dr = 0
            while ze <= z_max:

                # Intersect shape
                if shape == 'prism':
                    dr, _, _, _, = intersectBBox(ox, oy, oz, dx, dy, dz, scale_x, scale_y, scale_z)
                # Intersect bounding box walls
                _, xe, ye, ze = intersectBBox(ox, oy, oz, dx, dy, dz, bbox_sizex, bbox_sizey, 1e6)
                if ze <= z_max:  # intersection below object height -> record path length and cycle ray
                    path_length = np.append(path_length, dr)

                    ox = xe
                    oy = ye
                    oz = ze
                    if abs(ox-0.5*bbox_sizex) < kEpsilon:  # hit +x wall
                        ox = ox - bbox_sizex + kEpsilon
                    elif abs(ox+0.5*bbox_sizex) < kEpsilon:  # hit -x wall
                        ox = ox + bbox_sizex - kEpsilon
                    if abs(oy-0.5*bbox_sizey) < kEpsilon:  # hit +y wall
                        oy = oy - bbox_sizey + kEpsilon
                    elif abs(oy + 0.5 * bbox_sizey) < kEpsilon:  # hit -y wall
                        oy = oy + bbox_sizey - kEpsilon
            path_length[i+j*N] = dr
    if outputfile != '':
        np.savetxt(outputfile, path_length, delimiter=',')
    return path_length[path_length > kEpsilon]

def pathlengthdistribution(
    shape,
    scale_x,
    scale_y,
    scale_z,
    ray_zenith,
    ray_azimuth,
    nrays,
    plyfile="",
    bins=10,
    normalize=True,
):
    pl = pathlengths(shape, scale_x, scale_y, scale_z, ray_zenith, ray_azimuth, nrays, plyfile)
    hist, bin_edges = np.histogram(pl, bins=bins, density=normalize)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {"hist": hist, "bin_centers": bin_centers}

# ---------- Binomial radiation ----------
def compute_binomial_prism(
    sr, #Row spacing (meters)
    sza, # sun zenith angle (radians)
    psi, # sun azimuth relative to row orientation (radians)
    lai, # Leaf Area Index
    ameanv, # Leaf absorptivity in the visible (PAR) band
    ameann, # Leaf absorptivity in the near infra-red band (NIR) band
    rsoilv, # Soil reflectivity in the visible (PAR) band
    rsoiln,  # Soil reflectivity in the near infra-red band (NIR) band
    Srad_dir,  # Direct-beam incoming radiation (W m-2)
    Srad_diff,  # Diffuse incoming radiation (W m-2)
    fvis, # Fraction incoming radiation in the visible part of the spectrum
    fnir,  # Fraction incoming radiation in the near infra-red part of the spectrum
    CanopyHeight,# Crown height (meters)
    wc, # Canopy width (meters)
    sp, #Plant spacing (meters)
    Gtheta, # fraction of leaf area projected in the direction of the sun
    nrays,
    Nbins,
    shape="prism",
    Nz_diff=16,
    Nphi_diff=32,
    scattering=True
):
    """
    Returns:
        Rs_dir:  soil absorbed radiation - direct beam (W m-2)
        Rs_diff: soil absorbed radiation - diffuse (W m-2)
        Rc_dir:  canopy absorbed radiation - direct beam (W m-2)
        Rc_diff: canopy absorbed radiation - diffuse (W m-2)

    References:
    - Bailey, B.N., Ponce de León, M.A., and Krayenhoff, E.S., 2020. One-dimensional models of radiation transfer in homogeneous canopies: A review, re-evaluation, and improved model. Geoscientific Model Development 13:4789:4808
    - Bailey, B.N. and Fu, K., 2022. The probability distribution of absorbed direct, diffuse, and scattered radiation in plant canopies with varying structure. Agricultural and Forest Meteorology, 322, p.109009.
    - Ponce de León, M.A., Alfieri, J.G., Prueger, J.H., Hipps, L., Kustas, W.P., Agam, N., Bambach, N., McElrone, A.J., Knipper, K., Roby, M.C. and Bailey, B.N., 2025.
      One-dimensional modeling of radiation absorption by vine canopies: evaluation of existing model assumptions, and development of an improved generalized model.
      Agricultural and Forest Meteorology, 373, p.110706 (https://doi.org/10.1016/j.agrformet.2025.110706)
    - Path length distribution code: https://github.com/PlantSimulationLab/pathlengthdistribution
    """

    IncPAR_dir  = Srad_dir * fvis
    IncNIR_dir  = Srad_dir * fnir
    IncPAR_diff = Srad_diff * fvis
    IncNIR_diff = Srad_diff * fnir

    # Within-prism leaf area density
    a = lai * sr * sp / (sp * wc * CanopyHeight)

    # ---- Direct beam ----
    # Adjusted effective spacing for row-oriented canopies
    s = sr * np.sin(psi) ** 2 + sp * np.cos(psi) ** 2
    s2 = s ** 2

    # Area of a single prism shadow at solar zenith of 0
    S0 = sp * wc

    # Area of a single prism shadow at solar zenith (sza) and relative azimuth (psi)
    Stheta = (
        sp * wc
        + sp * CanopyHeight * np.tan(sza) * np.abs(np.sin(psi))
        + wc * CanopyHeight * np.tan(sza) * np.abs(np.cos(psi))
    )

    # Number of prisms intersected by a beam of radiation
    N_crown = Stheta / S0

    # Path length distribution for direct beam
    dist = pathlengthdistribution(
        shape=shape,
        scale_x=wc,
        scale_y=sp,
        scale_z=CanopyHeight,
        ray_zenith=sza,
        ray_azimuth=psi,
        nrays=int(nrays),
        bins=int(Nbins),
    )
    N = dist["hist"] / np.sum(dist["hist"])
    S = dist["bin_centers"]

    # Probability of intersecting a leaf within a prism (first and second order scattering)
    PlOne_PAR = np.sum(N * (1.0 - np.exp(-Gtheta * ameanv * a * S)))
    PlOne_NIR = np.sum(N * (1.0 - np.exp(-Gtheta * ameann * a * S)))
    PlTwo_PAR = np.sum(N * (1.0 - np.exp(-Gtheta * 2.0 * ameanv * a * S)))
    PlTwo_NIR = np.sum(N * (1.0 - np.exp(-Gtheta * 2.0 * ameann * a * S)))

    S0_over_s2 = S0 / s2

    # Canopy-level probability of interception
    Pc1_PAR = (s2 / (sr * sp)) * (1.0 - max(0.0, 1.0 - PlOne_PAR * S0_over_s2) ** N_crown)
    Pc1_NIR = (s2 / (sr * sp)) * (1.0 - max(0.0, 1.0 - PlOne_NIR * S0_over_s2) ** N_crown)
    Pc2_PAR = (s2 / (sr * sp)) * (1.0 - max(0.0, 1.0 - PlTwo_PAR * S0_over_s2) ** N_crown)
    Pc2_NIR = (s2 / (sr * sp)) * (1.0 - max(0.0, 1.0 - PlTwo_NIR * S0_over_s2) ** N_crown)

    # Soil term: fraction of transmitted radiation reflected by soil and re-intercepted by canopy
    soil_term_PAR = (1.0 - Pc1_PAR) * rsoilv * S0_over_s2 * PlOne_PAR
    soil_term_NIR = (1.0 - Pc1_NIR) * rsoiln * S0_over_s2 * PlOne_NIR

    canopy_frac_PAR = min(Pc2_PAR + soil_term_PAR, 1.0 - (1.0 - Pc1_PAR) * (1.0 - rsoilv))
    canopy_frac_NIR = min(Pc2_NIR + soil_term_NIR, 1.0 - (1.0 - Pc1_NIR) * (1.0 - rsoiln))

    # Direct radiation absorbed by the soil: transmitted fraction times soil absorptivity
    Rs_dir = (IncPAR_dir * (1.0 - Pc1_PAR) * (1.0 - rsoilv) +
              IncNIR_dir * (1.0 - Pc1_NIR) * (1.0 - rsoiln))

    # Direct radiation absorbed by the canopy
    if scattering:
        Rc_dir = canopy_frac_PAR * IncPAR_dir + canopy_frac_NIR * IncNIR_dir

    else:
        Rc_dir = Pc1_PAR * IncPAR_dir + Pc1_NIR * IncNIR_dir

    # ---- Diffuse sky: integrate over hemisphere ----
    dtheta = (0.5 * np.pi) / Nz_diff
    dphi   = (2.0 * np.pi) / Nphi_diff

    # Accumulators
    Pc1_PAR_diff      = 0.0
    Pc1_NIR_diff      = 0.0
    Pc2_PAR_diff      = 0.0
    Pc2_NIR_diff      = 0.0
    soil_term_PAR_diff = 0.0
    soil_term_NIR_diff = 0.0

    nrays_d = max(100, int(nrays // (Nz_diff * Nphi_diff)))

    for i in range(Nz_diff):
        theta     = (i + 0.5) * dtheta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        for j in range(Nphi_diff):
            phi   = (j + 0.5) * dphi
            w_dir = (cos_theta * sin_theta * dtheta * dphi) / np.pi

            # Path length distribution for this diffuse direction
            dist_d = pathlengthdistribution(
                shape=shape,
                scale_x=wc,
                scale_y=sp,
                scale_z=CanopyHeight,
                ray_zenith=theta,
                ray_azimuth=phi,
                nrays=nrays_d,
                bins=int(Nbins),
            )
            N_d = dist_d["hist"] / np.sum(dist_d["hist"])
            S_d = dist_d["bin_centers"]

            # Per-crown leaf interception probabilities for this direction
            PlOne_PAR_d = np.sum(N_d * (1.0 - np.exp(-Gtheta * ameanv * a * S_d)))
            PlOne_NIR_d = np.sum(N_d * (1.0 - np.exp(-Gtheta * ameann * a * S_d)))
            PlTwo_PAR_d = np.sum(N_d * (1.0 - np.exp(-Gtheta * 2.0 * ameanv * a * S_d)))
            PlTwo_NIR_d = np.sum(N_d * (1.0 - np.exp(-Gtheta * 2.0 * ameann * a * S_d)))

            # Direction-specific geometry
            Sthetadiff = (
                sp * wc
                + sp * CanopyHeight * np.tan(theta) * np.abs(np.sin(phi))
                + wc * CanopyHeight * np.tan(theta) * np.abs(np.cos(phi))
            )
            N_crowndiff  = Sthetadiff / S0
            s_d          = sr * np.sin(phi) ** 2 + sp * np.cos(phi) ** 2
            s2_d         = s_d ** 2
            S0_over_s2_d = S0 / s2_d

            # Per-direction canopy-level interception probabilities.
            Pc1_PAR_d = (s2_d / (sr * sp)) * (1.0 - max(0.0, 1.0 - PlOne_PAR_d * S0_over_s2_d) ** N_crowndiff)
            Pc1_NIR_d = (s2_d / (sr * sp)) * (1.0 - max(0.0, 1.0 - PlOne_NIR_d * S0_over_s2_d) ** N_crowndiff)
            Pc2_PAR_d = (s2_d / (sr * sp)) * (1.0 - max(0.0, 1.0 - PlTwo_PAR_d * S0_over_s2_d) ** N_crowndiff)
            Pc2_NIR_d = (s2_d / (sr * sp)) * (1.0 - max(0.0, 1.0 - PlTwo_NIR_d * S0_over_s2_d) ** N_crowndiff)

            # Per-direction soil scattering terms
            soil_term_PAR_d = (1.0 - Pc1_PAR_d) * rsoilv * S0_over_s2_d * PlOne_PAR_d
            soil_term_NIR_d = (1.0 - Pc1_NIR_d) * rsoiln * S0_over_s2_d * PlOne_NIR_d

            # Accumulate weighted contributions
            Pc1_PAR_diff       += w_dir * Pc1_PAR_d
            Pc1_NIR_diff       += w_dir * Pc1_NIR_d
            Pc2_PAR_diff       += w_dir * Pc2_PAR_d
            Pc2_NIR_diff       += w_dir * Pc2_NIR_d
            soil_term_PAR_diff += w_dir * soil_term_PAR_d
            soil_term_NIR_diff += w_dir * soil_term_NIR_d

    # Diffuse radiation absorbed by the soil
    Rs_diff = (IncPAR_diff * (1.0 - Pc1_PAR_diff) * (1.0 - rsoilv) +
               IncNIR_diff * (1.0 - Pc1_NIR_diff) * (1.0 - rsoiln))

    canopy_frac_PAR_diff = min(Pc2_PAR_diff + soil_term_PAR_diff,
                               1.0 - (1.0 - Pc1_PAR_diff) * (1.0 - rsoilv))
    canopy_frac_NIR_diff = min(Pc2_NIR_diff + soil_term_NIR_diff,
                               1.0 - (1.0 - Pc1_NIR_diff) * (1.0 - rsoiln))
    # Diffuse radiation absorbed by the canopy
    if scattering:
        Rc_diff = canopy_frac_PAR_diff * IncPAR_diff + canopy_frac_NIR_diff * IncNIR_diff

    else:
        Rc_diff = Pc1_PAR_diff * IncPAR_diff + Pc1_NIR_diff * IncNIR_diff

    Rs=Rs_dir + Rs_diff
    Rc=Rc_dir + Rc_diff
    albedo=Srad_dir+Srad_diff-Rs-Rc
    return Rc, Rs, albedo