import numpy as np

class LVGeometry:
    """
    Simplified ellipsoidal LV geometry for DCM parameter sweep.
    
    The LV is modeled as a prolate ellipsoid divided into 5 regions:
    basal_septal, mid_septal, apical, mid_lateral, basal_lateral
    
    Each region has a local radius of curvature and wall thickness,
    from which wall stress is computed via the Laplace relation.
    
    Parameters
    ----------
    long_axis : float
        Base-to-apex length in cm. Normal ~9cm, DCM ~12cm
    short_axis : float  
        Equatorial diameter in cm. Normal ~5cm, DCM ~7cm
    thickness : dict
        Wall thickness per region in cm.
        Keys: basal_septal, mid_septal, apical, mid_lateral, basal_lateral
        Normal ~1.0cm, DCM thinned to 0.5-0.7cm
    lv_pressure : float
        LV systolic pressure in mmHg. Default 120.
    """
    
    REGIONS = ['basal_septal', 'mid_septal', 'apical', 
               'mid_lateral', 'basal_lateral']
    
    # Parametric positions on ellipsoid for each region
    # phi = angle from long axis (0=pole/apex, pi/2=equator)
    # For prolate ellipsoid: apex is at pole (phi~0), base at equator (phi~pi/2)
    REGION_PHI = {
        'apical':        np.pi * 0.08,   # near pole, tight curvature
        'mid_septal':    np.pi * 0.30,
        'mid_lateral':   np.pi * 0.35,
        'basal_septal':  np.pi * 0.55,
        'basal_lateral': np.pi * 0.60,
    }

    def __init__(self, long_axis, short_axis, thickness, lv_pressure=120.0):
        self.a = long_axis / 2      # semi-major axis
        self.b = short_axis / 2     # semi-minor axis
        self.h = thickness          # dict: region -> thickness (cm)
        self.P = lv_pressure * 0.1333  # convert mmHg to kPa
        
        assert set(thickness.keys()) == set(self.REGIONS), \
            f"thickness must have keys: {self.REGIONS}"


    def regional_radius(self, region):
        """
        Mean radius of curvature at region on prolate ellipsoid surface.
        Semi-major axis a (long), semi-minor axis b (short).
        phi=0 at apex (pole), phi=pi/2 at equator.
        """
        phi = self.REGION_PHI[region]
        
        # For prolate ellipsoid with a=semi-major, b=semi-minor:
        # R1 (meridional) = (b^2/a) / (1 - e^2*cos^2(phi))^(3/2)  -- wrong frame
        # Use direct formula:
        sin2 = np.sin(phi)**2
        cos2 = np.cos(phi)**2
        
        # Denominator term
        D = np.sqrt(self.a**2 * sin2 + self.b**2 * cos2)
        
        # R1 meridional radius
        R1 = (self.a**2 * self.b**2) / D**3
        
        # R2 circumferential radius  
        R2 = (self.b**2) / D
        
        return (R1 + R2) / 2
    
    def wall_stress(self, region):
        """
        Regional wall stress via Laplace: sigma = P * r / (2 * h)
        Units: kPa
        """
        r = self.regional_radius(region)
        h = self.h[region]
        return (self.P * r) / (2 * h)
    
    def stress_map(self):
        """Returns dict of region -> wall stress (kPa)"""
        return {r: self.wall_stress(r) for r in self.REGIONS}


    def stretch_map(self):
        """
        Convert regional wall stress to diastolic stretch ratios.
        
        Uses a fixed reference stress (normal LV mean) for normalization
        so that DCM geometries show absolute increase in stretch.
        Apex excluded — geometric singularity in ellipsoid approximation.
        
        Reference: normal LV mean wall stress ~15 kPa (Grossman 1975)
        DCM diastolic stretch range: 1.05-1.25 (Kohl et al. 1999)
        """
        NORMAL_REFERENCE_STRESS = 15.0  # kPa, normal LV mean
        STRETCH_SENSITIVITY = 0.008     # stretch units per kPa above reference
        
        non_apical = [r for r in self.REGIONS if r != 'apical']
        stretch = {}
        for r in non_apical:
            stress = self.wall_stress(r)
            # Linear mapping: normal stress -> stretch ~1.0
            # elevated DCM stress -> stretch up to ~1.25
            s = 1.0 + STRETCH_SENSITIVITY * (stress - NORMAL_REFERENCE_STRESS)
            stretch[r] = max(1.0, min(1.3, s))  # clamp to physiological range
        
        return stretch

    def heterogeneity_index(self):
        """
        Coefficient of variation of wall stress across non-apical regions.
        Apex excluded — geometric outlier that dominates CV regardless of remodeling.
        """
        non_apical = [r for r in self.REGIONS if r != 'apical']
        stresses = [self.wall_stress(r) for r in non_apical]
        return np.std(stresses) / np.mean(stresses)

    def septal_lateral_gradient(self):
        """
        Ratio of mean septal stress to mean lateral stress.
        The key arrhythmogenic metric for asymmetric DCM.
        >1.0 = septal stress dominant, <1.0 = lateral stress dominant.
        """
        septal = np.mean([self.wall_stress(r) for r in 
                        ['basal_septal', 'mid_septal']])
        lateral = np.mean([self.wall_stress(r) for r in 
                        ['mid_lateral', 'basal_lateral']])
        return septal / lateral


def normal_lv():
    """Reference normal LV geometry"""
    return LVGeometry(
        long_axis=9.0,
        short_axis=5.0,
        thickness={r: 1.0 for r in LVGeometry.REGIONS},
        lv_pressure=120.0
    )

def dcm_uniform(severity=1.0):
    """
    Uniformly dilated DCM — same EF reduction, symmetric remodeling.
    severity: 0=normal, 1=severe DCM
    """
    long_axis = 9.0 + severity * 3.0     # up to 12cm
    short_axis = 5.0 + severity * 2.0    # up to 7cm
    thickness = {r: 1.0 - severity * 0.4 for r in LVGeometry.REGIONS}
    return LVGeometry(long_axis, short_axis, thickness)

def dcm_asymmetric(severity=1.0, asymmetry=0.5):
    """
    Asymmetrically remodeled DCM — septal thinning > lateral thinning.
    This is the key geometry for arrhythmia substrate.
    asymmetry: 0=uniform thinning, 1=septal wall half as thick as lateral
    """
    long_axis = 9.0 + severity * 3.0
    short_axis = 5.0 + severity * 2.0
    
    base_thickness = 1.0 - severity * 0.3
    septal_penalty = asymmetry * 0.3
    
    thickness = {
        'basal_septal':  base_thickness - septal_penalty,
        'mid_septal':    base_thickness - septal_penalty,
        'apical':        base_thickness - septal_penalty * 0.5,
        'mid_lateral':   base_thickness,
        'basal_lateral': base_thickness,
    }
    return LVGeometry(long_axis, short_axis, thickness)


if __name__ == '__main__':
    cases = [
        ('Normal LV', normal_lv()),
        ('DCM Uniform (severe)', dcm_uniform(severity=1.0)),
        ('DCM Asymmetric (severe, high asymmetry)', dcm_asymmetric(severity=1.0, asymmetry=1.0)),
    ]
    
    for name, lv in cases:
        print(f'=== {name} ===')
        smap = lv.stress_map()
        for region, stress in smap.items():
            print(f'  {region:<20} stress={stress:.1f} kPa')
        print(f'  Heterogeneity index:      {lv.heterogeneity_index():.4f}')
        print(f'  Septal/lateral gradient:  {lv.septal_lateral_gradient():.4f}')
        print()
