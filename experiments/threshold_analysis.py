import sys
sys.path.insert(0, '/Users/aru/Desktop/dcm')

import numpy as np
import matplotlib.pyplot as plt
from models.geometry import dcm_asymmetric
from models.single_cell import run_single_cell_metrics

# ----------------------------------------------------------------------
# Substrate vs geometry, at fixed severity = 1.0.
#
# Primary readout: regional resting (diastolic) membrane potential
# dispersion, the phase-correct substrate for diastolically engaged SACs.
# Secondary: APD90 dispersion.
#
# We deliberately do NOT impose a 5 ms or 10 ms "arrhythmia threshold."
# The literature does not support an absolute-ms dispersion threshold for
# reentry (Laurita & Rosenbaum 2000 report a structure-dependent 3.2 ms/mm
# spatial gradient; Akar & Rosenbaum 2003 a >10 ms/mm transmural gradient;
# Coronel et al. 2009 argue dispersion magnitude alone is not arrhythmogenic).
# Without electrotonic coupling this single-cell model produces a SUBSTRATE
# PROXY, not a reentry threshold. We therefore report the continuous,
# monotonic relationship between the septal/lateral wall-stress gradient and
# the substrate, and leave any clinical cut-point to prospective validation.
#
# The septal/lateral gradient equals the lateral/septal wall-thickness ratio
# exactly (matched longitudinal levels), so it is directly readable from a
# standard echo.
# ----------------------------------------------------------------------


def substrate(lv, baseline_vrest):
    sm = lv.stretch_map()
    apd, vrest = [], []
    for region, lam in sm.items():
        m = run_single_cell_metrics(stretch=lam)
        apd.append(m['apd90'])
        vrest.append(m['v_rest'])
    return {
        'vrest_range_mV': max(vrest) - min(vrest),
        'vrest_depol_mV': max(vrest) - baseline_vrest,
        'apd_range_ms': max(apd) - min(apd),
        'sl_gradient': lv.septal_lateral_gradient(),
    }


def run_threshold():
    baseline_vrest = run_single_cell_metrics(stretch=1.0)['v_rest']
    asymmetries = np.linspace(0.0, 1.0, 21)

    rows = []
    print('Substrate vs asymmetry (severity = 1.0)')
    print('%10s %12s %14s %14s %12s' %
          ('Asymm', 'S/L grad', 'Vrest range', 'Vrest depol', 'APD range'))
    print('%10s %12s %14s %14s %12s' %
          ('', '(= h_lat/h_sep)', '(mV)', '(mV)', '(ms)'))
    print('-' * 64)
    for asym in asymmetries:
        lv = dcm_asymmetric(severity=1.0, asymmetry=asym)
        m = substrate(lv, baseline_vrest)
        rows.append({'asymmetry': float(asym), **m})
        print('%10.2f %12.4f %14.3f %14.3f %12.3f' %
              (asym, m['sl_gradient'], m['vrest_range_mV'],
               m['vrest_depol_mV'], m['apd_range_ms']))

    grad = np.array([r['sl_gradient'] for r in rows])
    vr = np.array([r['vrest_range_mV'] for r in rows])
    # Monotonic relationship (deterministic model output; report as a
    # relationship and a slope, NOT as an inferential statistic with a p-value)
    slope, intercept = np.polyfit(grad, vr, 1)
    r_val = np.corrcoef(grad, vr)[0, 1]
    print()
    print('Resting-Vm dispersion vs septal/lateral gradient:')
    print('  slope = %.3f mV per unit gradient   (linear fit r = %.3f)'
          % (slope, r_val))
    print('  i.e. each 0.1 rise in the echo thickness-ratio gradient adds')
    print('  ~%.2f mV of regional resting-potential dispersion.' % (slope * 0.1))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Asymmetric remodeling and the diastolic-depolarization substrate',
                 fontsize=12, fontweight='bold')

    x = [r['asymmetry'] for r in rows]
    vr_l = [r['vrest_range_mV'] for r in rows]
    vd_l = [r['vrest_depol_mV'] for r in rows]
    apd_l = [r['apd_range_ms'] for r in rows]

    ax = axes[0]
    ax.plot(x, vd_l, 'o-', color='darkred', linewidth=2, markersize=5,
            label='Diastolic depolarization, septum vs baseline (mV)')
    ax.plot(x, vr_l, 's-', color='crimson', linewidth=2, markersize=4,
            label='Resting Vm dispersion, range (mV)')
    ax2 = ax.twinx()
    ax2.plot(x, apd_l, '^--', color='steelblue', linewidth=1.5, markersize=4,
             label='APD90 dispersion (ms)')
    ax.set_xlabel('Remodeling asymmetry index')
    ax.set_ylabel('Resting membrane potential change (mV)')
    ax2.set_ylabel('APD90 dispersion (ms)', color='steelblue')
    ax.set_title('A. Substrate vs asymmetry')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    ax = axes[1]
    ax.scatter(grad, vr, c=x, cmap='plasma', s=55, zorder=3)
    xline = np.linspace(grad.min(), grad.max(), 100)
    ax.plot(xline, slope * xline + intercept, 'k--', alpha=0.5)
    ax.text(0.05, 0.92, 'slope %.2f mV per unit gradient' % slope,
            transform=ax.transAxes, fontsize=9)
    ax.set_xlabel('Septal/lateral stress gradient  (= lateral/septal wall-thickness ratio)')
    ax.set_ylabel('Resting Vm dispersion (mV)')
    ax.set_title('B. Substrate vs echo-computable gradient')

    plt.tight_layout()
    plt.savefig('figures/threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('Figure saved to figures/threshold_analysis.png')
    return rows


if __name__ == '__main__':
    run_threshold()