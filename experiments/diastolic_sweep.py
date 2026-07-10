import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from models.geometry import normal_lv, dcm_asymmetric
from models.single_cell import run_single_cell_metrics

# ----------------------------------------------------------------------
# Head-to-head readout comparison.
#
# Both readouts are computed from the SAME single-cell runs for each
# geometry:
#   1. APD90 dispersion (the original, plateau-phase readout)
#   2. Resting/diastolic membrane potential dispersion (the diastolic
#      readout that matches the phase in which SACs engage)
#
# The diastolic SAC current is inward at rest, so higher diastolic stretch
# depolarises the resting potential. Regional stretch heterogeneity should
# therefore produce a regional resting-Vm gradient, a direct excitability
# substrate. This script reports both so the two can be compared.
# ----------------------------------------------------------------------


def geometry_metrics(lv, baseline_vrest):
    stretch_map = lv.stretch_map()
    apd, vrest = {}, {}
    for region, lam in stretch_map.items():
        m = run_single_cell_metrics(stretch=lam)
        apd[region] = m['apd90']
        vrest[region] = m['v_rest']

    apd_vals = list(apd.values())
    vr_vals = list(vrest.values())
    return {
        'apd_range_ms': max(apd_vals) - min(apd_vals),
        'apd_cv': float(np.std(apd_vals) / np.mean(apd_vals)),
        # resting Vm is negative; report dispersion as a range and the
        # depolarisation of the most-stretched region vs the no-SAC baseline
        'vrest_range_mV': max(vr_vals) - min(vr_vals),
        'vrest_depol_mV': max(vr_vals) - baseline_vrest,
        'stress_heterogeneity': lv.heterogeneity_index(),
        'septal_lateral_gradient': lv.septal_lateral_gradient(),
        'apd_map': apd,
        'vrest_map': vrest,
    }


def run_sweep():
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    asymmetries = [0.0, 0.25, 0.5, 0.75, 1.0]

    # No-SAC baseline resting Vm (stretch 1.0 -> zero SAC conductance)
    baseline_vrest = run_single_cell_metrics(stretch=1.0)['v_rest']
    print('Baseline resting Vm (no SAC): %.2f mV' % baseline_vrest)
    print()
    print('%9s %9s %12s %12s %14s %10s' %
          ('Severity', 'Asymm', 'APD range', 'Vrest range', 'Vrest depol',
           'S/L grad'))
    print('%9s %9s %12s %12s %14s %10s' %
          ('', '', '(ms)', '(mV)', '(mV)', ''))
    print('-' * 70)

    results = []
    for severity in severities:
        for asymmetry in asymmetries:
            lv = normal_lv() if severity == 0.0 else \
                dcm_asymmetric(severity=severity, asymmetry=asymmetry)
            m = geometry_metrics(lv, baseline_vrest)
            results.append({'severity': severity, 'asymmetry': asymmetry, **m})
            print('%9.2f %9.2f %12.3f %12.3f %14.3f %10.4f' %
                  (severity, asymmetry, m['apd_range_ms'],
                   m['vrest_range_mV'], m['vrest_depol_mV'],
                   m['septal_lateral_gradient']))

    with open('data/diastolic_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to data/diastolic_sweep_results.json')
    return results


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    results = run_sweep()

    def pick(sev, asym):
        return [r for r in results
                if r['severity'] == sev and r['asymmetry'] == asym][0]

    normal = [r for r in results if r['severity'] == 0.0][0]
    uniform = pick(1.0, 0.0)
    asymm = pick(1.0, 1.0)

    print('\n=== Head to head (severity 1.0) ===')
    print('%-16s %14s %16s' % ('', 'APD range (ms)', 'Vrest range (mV)'))
    for name, r in [('Normal', normal), ('Uniform DCM', uniform),
                    ('Asymmetric DCM', asymm)]:
        print('%-16s %14.3f %16.3f'
              % (name, r['apd_range_ms'], r['vrest_range_mV']))
