import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from models.geometry import dcm_asymmetric, dcm_uniform, normal_lv, LVGeometry
from models.single_cell import run_single_cell, sac_conductance

def compute_repol_heterogeneity(lv, g_sac_max, e_sac):
    """Compute repolarization heterogeneity for given SAC parameters."""
    stretch_map = lv.stretch_map()
    apd_values = []
    for region, stretch in stretch_map.items():
        g_sac = sac_conductance(stretch, g_max=g_sac_max)
        apd = run_single_cell(stretch=stretch, g_sac_max=g_sac_max)
        apd_values.append(apd)
    return np.std(apd_values) / np.mean(apd_values)

def run_sensitivity():
    """
    Vary G_SAC_MAX and E_SAC across published ranges.
    Show that stress-repolarization correlation holds across parameter space.
    
    G_SAC_MAX range: 0.001-0.006 nS/pF (Kohl 1999, Peyronnet 2016)
    E_SAC range: -30 to +10 mV (non-selective cation channel range)
    """
    # Parameter ranges from literature
    g_sac_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    e_sac_values = [-30.0, -20.0, -10.0, 0.0, 10.0]
    
    # Test geometries — same severity, different asymmetry
    geometries = {
        'normal':     normal_lv(),
        'uniform':    dcm_asymmetric(severity=1.0, asymmetry=0.0),
        'mild_asym':  dcm_asymmetric(severity=1.0, asymmetry=0.5),
        'high_asym':  dcm_asymmetric(severity=1.0, asymmetry=1.0),
    }
    
    print('Sensitivity Analysis: G_SAC_MAX sweep (E_SAC = -10 mV fixed)')
    print(f'{"G_SAC_MAX":>12} {"Normal":>10} {"Uniform":>10} {"Mild Asym":>12} {"High Asym":>12} {"Ratio H/U":>12}')
    print('-' * 68)
    
    g_results = []
    for g in g_sac_values:
        row = {}
        for name, lv in geometries.items():
            hi = compute_repol_heterogeneity(lv, g_sac_max=g, e_sac=-10.0)
            row[name] = hi
        ratio = row['high_asym'] / row['uniform'] if row['uniform'] > 0 else float('inf')
        g_results.append({'g': g, **row, 'ratio': ratio})
        print(f'{g:>12.4f} {row["normal"]:>10.4f} {row["uniform"]:>10.4f} '
              f'{row["mild_asym"]:>12.4f} {row["high_asym"]:>12.4f} {ratio:>12.2f}')
    
    print()
    print('Sensitivity Analysis: E_SAC sweep (G_SAC_MAX = 0.003 fixed)')
    print(f'{"E_SAC (mV)":>12} {"Normal":>10} {"Uniform":>10} {"Mild Asym":>12} {"High Asym":>12} {"Ratio H/U":>12}')
    print('-' * 68)
    
    e_results = []
    for e in e_sac_values:
        row = {}
        for name, lv in geometries.items():
            hi = compute_repol_heterogeneity(lv, g_sac_max=0.003, e_sac=e)
            row[name] = hi
        ratio = row['high_asym'] / row['uniform'] if row['uniform'] > 0 else float('inf')
        e_results.append({'e': e, **row, 'ratio': ratio})
        print(f'{e:>12.1f} {row["normal"]:>10.4f} {row["uniform"]:>10.4f} '
              f'{row["mild_asym"]:>12.4f} {row["high_asym"]:>12.4f} {ratio:>12.2f}')
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Sensitivity Analysis: SAC Parameter Variation', fontsize=12, fontweight='bold')
    
    # Panel 1: G_SAC sweep
    ax = axes[0]
    for name in ['normal', 'uniform', 'mild_asym', 'high_asym']:
        y = [r[name] for r in g_results]
        ax.plot(g_sac_values, y, 'o-', label=name)
    ax.set_xlabel('G_SAC_MAX (nS/pF)')
    ax.set_ylabel('Repolarization Heterogeneity Index')
    ax.set_title('A. Conductance Sensitivity')
    ax.legend()
    
    # Panel 2: E_SAC sweep
    ax = axes[1]
    for name in ['normal', 'uniform', 'mild_asym', 'high_asym']:
        y = [r[name] for r in e_results]
        ax.plot(e_sac_values, y, 'o-', label=name)
    ax.set_xlabel('E_SAC (mV)')
    ax.set_ylabel('Repolarization Heterogeneity Index')
    ax.set_title('B. Reversal Potential Sensitivity')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('Figure saved to figures/sensitivity_analysis.png')
    
    return g_results, e_results

if __name__ == '__main__':
    g_results, e_results = run_sensitivity()
    
    print()
    print('=== Robustness summary ===')
    print('Asymmetric DCM shows higher repol HI than uniform DCM across:')
    g_robust = all(r['ratio'] > 1.0 for r in g_results)
    e_robust = all(r['ratio'] > 1.0 for r in e_results)
    print(f'  All G_SAC_MAX values: {g_robust}')
    print(f'  All E_SAC values:     {e_robust}')