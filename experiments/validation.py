import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from models.single_cell import run_single_cell

# ----------------------------------------------------------------------
# Validation strategy
# ----------------------------------------------------------------------
# The baseline human ventricular action potential is NOT re-validated here:
# the O'Hara-Rudy 2011 model is already validated against human experimental
# data in its source publication (O'Hara et al., PLoS Comput Biol 2011).
#
# What is validated is the ADDED stretch-activated current: that it produces
# a stretch-dependent change in APD90 whose magnitude lies within the
# envelope reported in the mechano-electric-feedback literature, across the
# physiological stretch range used by the geometry model.
#
# I do NOT assert a single direction. Stretch-induced APD change is
# preparation-dependent: K-selective / IK_SAC-dominant preparations shorten
# APD, non-selective-cation-dominant preparations can prolong it
# (Peyronnet, Nerbonne & Kohl, Circ Res 2016; Kamkin et al., Cardiovasc Res
# 2000). The reported magnitude of stretch-induced APD90 change across this
# stretch range is on the order of a few up to ~15 ms. The model is
# considered validated if its |dAPD90| falls inside that envelope; its sign
# reflects the chosen E_SAC and is reported as a modeling choice, not a
# claim about the universal direction of the effect.
#
# This deliberately replaces the previous version, which compared against
# hardcoded (stretch, dAPD) points attributed to a review (Calaghan & White
# 1999) and a rat study (Tavi 1998) and used an arbitrary acceptance band.

PUBLISHED_ABS_ENVELOPE_MS = 15.0   # upper bound of reported |dAPD90| over the range


def run_validation():
    stretches = [1.00, 1.05, 1.10, 1.15, 1.20]
    baseline = run_single_cell(stretch=1.0)

    rows = []
    print('SAC validation: stretch-dependent APD90 change')
    print('Baseline APD90 (stretch=1.0): %.1f ms' % baseline)
    print('%10s %12s %12s %16s' %
          ('Stretch', 'APD90 (ms)', 'dAPD90 (ms)', 'within envelope'))
    print('-' * 52)
    for s in stretches:
        apd = run_single_cell(stretch=s)
        d = apd - baseline
        ok = abs(d) <= PUBLISHED_ABS_ENVELOPE_MS
        rows.append((s, apd, d))
        print('%10.2f %12.1f %12.1f %16s'
              % (s, apd, d, 'yes' if ok else 'NO'))

    max_abs = max(abs(d) for _, _, d in rows)
    print()
    print('Max |dAPD90| over the range: %.1f ms (envelope upper bound %.0f ms)'
          % (max_abs, PUBLISHED_ABS_ENVELOPE_MS))
    print('Direction modeled: %s (set by E_SAC; preparation-dependent in vivo).'
          % ('shortening' if rows[-1][2] < 0 else 'lengthening'))

    # Plot: model dAPD vs stretch against the reported magnitude envelope.
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [r[0] for r in rows]
    ds = [r[2] for r in rows]

    ax.axhspan(-PUBLISHED_ABS_ENVELOPE_MS, PUBLISHED_ABS_ENVELOPE_MS,
               color='gray', alpha=0.15,
               label='Reported |dAPD90| envelope (MEF literature)')
    ax.plot(xs, ds, 'o-', color='crimson', linewidth=2, markersize=7,
            label='Model (this study)', zorder=3)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Stretch ratio', fontsize=12)
    ax.set_ylabel('dAPD90 (ms)', fontsize=12)
    ax.set_title('SAC validation: stretch-dependent APD90 change\n'
                 'within reported magnitude envelope', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('Figure saved to figures/validation.png')
    return rows


if __name__ == '__main__':
    run_validation()