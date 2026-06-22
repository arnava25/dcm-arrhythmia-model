import numpy as np

# ----------------------------------------------------------------------
# Module-level calibration
# ----------------------------------------------------------------------
# Normal LV reference dimensions, cm. (a, b) are treated as the
# END-DIASTOLIC ENDOCARDIAL surface throughout, so the ellipsoid cavity
# volume is the end-diastolic volume (EDV).
_NORMAL_LONG_AXIS = 9.0
_NORMAL_SHORT_AXIS = 5.0

# EF calibration. Stroke volume is held fixed at the value that gives the
# normal LV an EF of 0.60. EF then falls as the chamber dilates (EDV
# rises) and is, by construction, independent of wall-thickness
# asymmetry, because it depends only on (a, b). This is the formal basis
# of the "equivalent EF" comparison: at fixed severity, varying asymmetry
# leaves a, b unchanged, hence EF unchanged.
NORMAL_EF = 0.60

MMHG_TO_KPA = 0.13332

# Sarcomere working range (Sequeira et al., J Gen Physiol 2023): the passive
# working range spans ~1.9 to 2.3 um. Relative to a slack length of 1.85 um,
# the top of the range is a fiber stretch of 2.30/1.85 = 1.24. Diastolic
# stretch is capped here, NOT at 1.3 (the old Kohl-1999-attributed value,
# which the literature pass could not verify).
SL_SLACK = 1.85
SL_MAX = 2.30
LAMBDA_MAX = SL_MAX / SL_SLACK


def _ellipsoid_volume(a, b):
    """Cavity volume of a prolate ellipsoid, mL when a, b are in cm.
    Full-ellipsoid volume; any apex-to-base truncation factor cancels in
    EF and is omitted. For normal dimensions this gives ~118 mL, a
    physiological normal EDV."""
    return (4.0 / 3.0) * np.pi * a * b ** 2


_NORMAL_EDV = _ellipsoid_volume(_NORMAL_LONG_AXIS / 2, _NORMAL_SHORT_AXIS / 2)
FIXED_STROKE_VOLUME = NORMAL_EF * _NORMAL_EDV  # mL, constant across geometries


class LVGeometry:
    """Simplified prolate-ellipsoid LV geometry for the DCM parameter sweep.

    Five regions sit on the endocardial surface. Septal and lateral
    regions at the same longitudinal level share the SAME parametric
    angle phi, so the surface is axisymmetric and the septal/lateral
    stress gradient is exactly 1.0 under uniform wall thickness. Any
    gradient above 1.0 therefore arises only from differential wall
    thinning, not from the parametrization.

    Wall stress uses the thin-wall biaxial membrane equations for a shell
    of revolution (two principal components), not the single-component
    spherical Laplace law.
    """

    REGIONS = ['basal_septal', 'mid_septal', 'apical',
               'mid_lateral', 'basal_lateral']

    NON_APICAL = ['basal_septal', 'mid_septal', 'mid_lateral', 'basal_lateral']

    # phi measured from the apical pole: phi=0 at apex, phi=pi/2 at the
    # equatorial base. Matched septal/lateral levels.
    REGION_PHI = {
        'apical':        np.pi * 0.15,
        'mid_septal':    np.pi * 0.38,
        'mid_lateral':   np.pi * 0.38,
        'basal_septal':  np.pi * 0.50,
        'basal_lateral': np.pi * 0.50,
    }

    def __init__(self, long_axis, short_axis, thickness,
                 lv_systolic_pressure=120.0, lv_diastolic_pressure=8.0):
        self.a = long_axis / 2.0    # polar (long) semi-axis, endocardial
        self.b = short_axis / 2.0   # equatorial (short) semi-axis, endocardial
        self.h = dict(thickness)    # region -> wall thickness, cm
        self.P_sys = lv_systolic_pressure * MMHG_TO_KPA
        self.P_dia = lv_diastolic_pressure * MMHG_TO_KPA
        assert set(self.h.keys()) == set(self.REGIONS), \
            "thickness must have keys: %s" % self.REGIONS

    # --- curvature ----------------------------------------------------
    def principal_radii(self, region):
        """Meridional R_m and transverse (circumferential) R_t radii of
        curvature of the endocardial prolate ellipsoid at a region, cm.

        With a = polar semi-axis, b = equatorial semi-axis,
        D = sqrt(a^2 sin^2 phi + b^2 cos^2 phi):
            R_m = D^3 / (a b)
            R_t = b D / a
        Checks: at the pole (phi=0) both equal b^2/a (umbilic);
        at the equator (phi=pi/2) R_m = a^2/b and R_t = b.
        """
        phi = self.REGION_PHI[region]
        D = np.sqrt(self.a ** 2 * np.sin(phi) ** 2 + self.b ** 2 * np.cos(phi) ** 2)
        R_m = D ** 3 / (self.a * self.b)
        R_t = self.b * D / self.a
        return R_m, R_t

    # --- stress -------------------------------------------------------
    def wall_stress(self, region, component='circumferential', phase='systole'):
        """Thin-wall biaxial membrane stress, kPa.

        component = 'circumferential' (hoop, the conventional LV wall
        stress and the larger component) or 'meridional'.

            sigma_meridional      = P R_t / (2 h)
            sigma_circumferential = (P R_t / h) (1 - R_t / (2 R_m))

        Both the heterogeneity CV and the septal/lateral gradient are
        ratios, so they are invariant to the absolute pressure (sigma
        scales linearly with P). Only the absolute magnitude depends on
        phase, which matters for stretch, not for the geometric metrics.
        """
        P = self.P_sys if phase == 'systole' else self.P_dia
        R_m, R_t = self.principal_radii(region)
        h = self.h[region]
        sigma_m = P * R_t / (2.0 * h)
        if component == 'meridional':
            return sigma_m
        return (P * R_t / h) * (1.0 - R_t / (2.0 * R_m))

    def stress_map(self, component='circumferential', phase='systole'):
        return {r: self.wall_stress(r, component, phase) for r in self.REGIONS}

    # --- geometric heterogeneity metrics ------------------------------
    def heterogeneity_index(self):
        """Coefficient of variation of circumferential systolic wall
        stress across the four non-apical regions. Apex excluded (it is
        on-axis and its thickness rule is arbitrary). Pressure-invariant."""
        stresses = [self.wall_stress(r) for r in self.NON_APICAL]
        return float(np.std(stresses) / np.mean(stresses))

    def septal_lateral_gradient(self):
        """Mean septal / mean lateral circumferential wall stress.
        Exactly 1.0 under uniform thickness (matched phi); rises only with
        differential thinning. Pressure-invariant."""
        septal = np.mean([self.wall_stress(r) for r in ['basal_septal', 'mid_septal']])
        lateral = np.mean([self.wall_stress(r) for r in ['mid_lateral', 'basal_lateral']])
        return float(septal / lateral)

    # --- volume, EF, strain -------------------------------------------
    def cavity_volume(self):
        """End-diastolic cavity volume, mL."""
        return _ellipsoid_volume(self.a, self.b)

    def ejection_fraction(self):
        """EF with stroke volume fixed at the normal-LV calibration.
        Depends only on (a, b); invariant to wall-thickness asymmetry."""
        return FIXED_STROKE_VOLUME / self.cavity_volume()

    def global_strain(self):
        """Isotropic linear shortening implied by EF: 1 - (1-EF)^(1/3).
        Falls with dilation, as in DCM."""
        ef = self.ejection_fraction()
        return 1.0 - (1.0 - ef) ** (1.0 / 3.0)

    # --- stretch (diastolic, SAC-engaging phase) ----------------------
    def stretch_map(self):
        """Regional END-DIASTOLIC fiber stretch driving SAC activation.

        SACs are engaged during diastolic filling, when lengthened myocytes
        at resting potential carry an inward non-selective SAC current
        (Peyronnet, Nerbonne & Kohl 2016); peak systole is the wrong phase.
        Stretch is therefore computed from end-diastolic circumferential
        wall stress at LVEDP.

        Reference: lambda = 1.0 at the NORMAL end-diastolic loaded state.
        Regions whose end-diastolic stress exceeds the normal mean are
        stretched proportionally further through a linear first-order
        passive compliance, capped at the sarcomere working-range ceiling
        LAMBDA_MAX (Sequeira et al. 2023).

        The spatial PATTERN of stretch (which regions stretch more) is fixed
        by geometry alone (curvature / thickness) and is independent of the
        compliance calibration; only the absolute magnitude depends on it.
        Apex excluded.

        Methods limitations to state: reference is the loaded normal
        diastolic state, not the unloaded stress-free configuration
        (residual stress ignored); the passive stress-strain relation is
        linearized; LVEDP is held at a fixed baseline, and elevated DCM
        filling pressure (which would amplify absolute stretch) is handled
        by sensitivity sweep, not built into the base case.
        """
        ref, C = _stretch_calibration()
        stretch = {}
        for r in self.NON_APICAL:
            s_ed = self.wall_stress(r, 'circumferential', phase='diastole')
            lam = 1.0 + C * (s_ed - ref)
            stretch[r] = float(np.clip(lam, 1.0, LAMBDA_MAX))
        return stretch


# ----------------------------------------------------------------------
# Reference geometries
# ----------------------------------------------------------------------
def normal_lv():
    return LVGeometry(_NORMAL_LONG_AXIS, _NORMAL_SHORT_AXIS,
                      {r: 1.0 for r in LVGeometry.REGIONS})


def dcm_asymmetric(severity=1.0, asymmetry=0.5):
    """DCM geometry. severity drives dilation and global thinning;
    asymmetry drives differential septal-vs-lateral thinning.

        long_axis  = 9.0 + 3.0 * severity        (cm)
        short_axis = 5.0 + 2.0 * severity         (cm)
        base_thickness = 1.0 - 0.3 * severity     (cm)
        septal_penalty = 0.3 * asymmetry          (cm)
        septal regions  = base_thickness - septal_penalty
        lateral regions = base_thickness
    At severity = asymmetry = 1: septal = 0.4 cm, lateral = 0.7 cm.
    """
    long_axis = _NORMAL_LONG_AXIS + severity * 3.0
    short_axis = _NORMAL_SHORT_AXIS + severity * 2.0
    base_thickness = 1.0 - severity * 0.3
    septal_penalty = asymmetry * 0.3
    thickness = {
        'basal_septal':  base_thickness - septal_penalty,
        'mid_septal':    base_thickness - septal_penalty,
        'apical':        base_thickness,
        'mid_lateral':   base_thickness,
        'basal_lateral': base_thickness,
    }
    return LVGeometry(long_axis, short_axis, thickness)


def dcm_uniform(severity=1.0):
    """Uniformly dilated DCM: the asymmetry = 0 case of dcm_asymmetric.
    Defined as a wrapper so the uniform comparator is identical to the
    sweep's asymmetry = 0 geometry (no separate, inconsistent thinning
    rule)."""
    return dcm_asymmetric(severity=severity, asymmetry=0.0)


# Diastolic stretch calibration, computed once. Reference = normal-LV mean
# end-diastolic circumferential stress (the normal diastolic loading).
# Compliance C is set so the most-stressed region of the most asymmetric
# severe DCM reaches LAMBDA_MAX, anchoring the absolute scale to the
# sarcomere ceiling. The spatial pattern is independent of C.
_STRETCH_CAL = None


def _stretch_calibration():
    global _STRETCH_CAL
    if _STRETCH_CAL is None:
        ref = float(np.mean([
            normal_lv().wall_stress(r, 'circumferential', phase='diastole')
            for r in LVGeometry.NON_APICAL]))
        worst = dcm_asymmetric(severity=1.0, asymmetry=1.0)
        s_max = max(worst.wall_stress(r, 'circumferential', phase='diastole')
                    for r in LVGeometry.NON_APICAL)
        C = (LAMBDA_MAX - 1.0) / (s_max - ref)
        _STRETCH_CAL = (ref, C)
    return _STRETCH_CAL


if __name__ == '__main__':
    print('=== curvature checks ===')
    lv = normal_lv()
    Rm0, Rt0 = lv.principal_radii('apical')  # not the pole, but near it
    # explicit pole/equator check
    a, b = lv.a, lv.b
    print('  a=%.3f  b=%.3f' % (a, b))
    print('  pole    R_m,R_t expected b^2/a=%.3f' % (b ** 2 / a))
    print('  equator R_m expected a^2/b=%.3f , R_t expected b=%.3f' % (a ** 2 / b, b))

    print()
    print('=== equator hoop vs meridional ratio (normal) ===')
    Rm, Rt = lv.principal_radii('basal_septal')  # phi=0.50pi = equator
    sm = lv.wall_stress('basal_septal', 'meridional')
    st = lv.wall_stress('basal_septal', 'circumferential')
    print('  meridional=%.2f kPa  circumferential=%.2f kPa  ratio=%.3f'
          % (sm, st, st / sm))

    print()
    print('=== reference cases ===')
    cases = [
        ('Normal LV', normal_lv()),
        ('DCM uniform (sev 1.0)', dcm_uniform(1.0)),
        ('DCM asym (sev 1.0, asym 0.5)', dcm_asymmetric(1.0, 0.5)),
        ('DCM asym (sev 1.0, asym 1.0)', dcm_asymmetric(1.0, 1.0)),
    ]
    print('%-32s %8s %8s %8s %8s %8s' %
          ('case', 'EDV', 'EF', 'strain', 'CV', 'S/L'))
    for name, g in cases:
        print('%-32s %8.1f %8.3f %8.3f %8.4f %8.4f' %
              (name, g.cavity_volume(), g.ejection_fraction(),
               g.global_strain(), g.heterogeneity_index(),
               g.septal_lateral_gradient()))

    print()
    print('=== EF invariance to asymmetry at fixed severity ===')
    for asym in [0.0, 0.25, 0.5, 0.75, 1.0]:
        g = dcm_asymmetric(severity=1.0, asymmetry=asym)
        print('  asym=%.2f  EF=%.4f  S/L=%.4f  CV=%.4f'
              % (asym, g.ejection_fraction(),
                 g.septal_lateral_gradient(), g.heterogeneity_index()))