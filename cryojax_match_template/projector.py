"""cryojax projector taking cisTEM-convention angles and CTF parameters."""
import cryojax.simulator as cxs


def make_proj_fn(volume, image_config, outputs_real_space=True):
    """Returns proj(phi, theta, psi, off_x_A, off_y_A, df1, df2, df_ang, Cs, amp).

    Angles are cisTEM ZYZ (degrees). cryojax's EulerAnglePose has phi and psi
    swapped relative to cisTEM, so they are swapped here.
    df_ang is in the units cryojax expects for astigmatism_angle.
    """
    def proj(phi_c, theta_c, psi_c, off_x_A, off_y_A, df1, df2, df_ang, Cs, amp):
        pose = cxs.EulerAnglePose(
            phi_angle=psi_c, theta_angle=theta_c, psi_angle=phi_c,
            offset_x_in_angstroms=off_x_A, offset_y_in_angstroms=off_y_A)
        ctf = cxs.AstigmaticCTF(defocus_in_angstroms=(df1 + df2) * 0.5,
                                astigmatism_in_angstroms=df1 - df2,
                                astigmatism_angle=df_ang,
                                spherical_aberration_in_mm=Cs)
        tt = cxs.ContrastTransferTheory(ctf=ctf, amplitude_contrast_ratio=amp)
        model = cxs.LinearImageModel(volume=volume, pose=pose,
                                     image_config=image_config, transfer_theory=tt)
        return model.simulate(outputs_real_space=outputs_real_space)
    return proj
