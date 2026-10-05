"""cisTEM EulerSearch orientation grid (as used by match_template)."""
import numpy as np


def symmetry_limits(symmetry):
    s = symmetry.strip().upper()
    if len(s) < 1:
        raise ValueError("empty symmetry")
    typ = s[0]
    n = int(s[1:]) if len(s) > 1 else 0
    if typ == "C":
        if n <= 0:
            raise ValueError("C symmetry must be like C1, C2, ...")
        return 360.0 / n, 90.0, True
    if typ == "D":
        if n <= 0:
            raise ValueError("D symmetry must be like D2, D7, ...")
        return 360.0 / n, 90.0, False
    if typ == "T":
        return 180.0, 54.7, False
    if typ == "O":
        return 90.0, 54.7, False
    if typ == "I":
        return 180.0, 31.7, False
    raise ValueError("unsupported symmetry {}".format(symmetry))


def cistem_sphere_grid(symmetry, angular_step, c_match_template=True):
    """Out-of-plane grid as (N, 2) [phi, theta] in degrees."""
    phi_max, theta_max, test_mirror = symmetry_limits(symmetry)
    if c_match_template and symmetry.strip().upper().startswith("C") and test_mirror:
        theta_max = 180.0

    # EulerSearch accumulates in C++ float. Matching float32 matters: at some
    # latitudes it includes one final phi sample that float64 would omit.
    phi_max = np.float32(phi_max)
    theta_max = np.float32(theta_max)
    angular_step = np.float32(angular_step)
    theta_step = np.float32(
        theta_max / int(theta_max / angular_step + np.float32(0.5))
    )
    rows = []
    theta = np.float32(0.0)
    while theta < np.float32(theta_max + theta_step / np.float32(2.0)):
        if theta == np.float32(0.0) or theta == np.float32(180.0):
            phi_step = phi_max
        else:
            phi_step = np.float32(abs(angular_step / np.sin(np.deg2rad(theta))))
            if phi_step > phi_max:
                phi_step = phi_max
            phi_step = np.float32(phi_max / int(phi_max / phi_step + np.float32(0.5)))
        phi = np.float32(0.0)
        while phi < phi_max:
            rows.append((phi, theta))
            phi = np.float32(phi + phi_step)
        theta = np.float32(theta + theta_step)
    return np.asarray(rows, dtype=np.float32), float(theta_step)


def cistem_pose_grid(symmetry, angular_step, psi_step, include_psi_360=True):
    """Full (phi, theta, psi) grid in cisTEM ZYZ convention, degrees.

    Returns (poses (N, 3), sphere (M, 2), psis (K,), theta_step).
    """
    sphere, theta_step = cistem_sphere_grid(symmetry, angular_step)
    psi_step = np.float32(psi_step)
    psi_max = np.float32(360.0) if include_psi_360 else np.float32(360.0) - psi_step
    psis = []
    psi = np.float32(0.0)
    while psi <= psi_max:
        psis.append(psi)
        psi = np.float32(psi + psi_step)
    psis = np.asarray(psis, dtype=np.float32)
    poses = np.empty((sphere.shape[0] * psis.shape[0], 3), dtype=np.float32)
    out = 0
    for phi, theta in sphere:
        n = psis.shape[0]
        poses[out:out + n, 0] = phi
        poses[out:out + n, 1] = theta
        poses[out:out + n, 2] = psis
        out += n
    return poses, sphere, psis, theta_step
