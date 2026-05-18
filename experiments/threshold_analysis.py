import sys
sys.path.insert(0, '/Users/aru/Desktop/dcm')

import numpy as np
import matplotlib.pyplot as plt
from models.geometry import dcm_asymmetric, normal_lv
from models.single_cell import run_single_cell

def compute_repol_heterogeneity(lv):
    stretch_map = lv.stretch_map()
    apd_values = []
    for region, stretch in stretch_map.items():
        apd = run_single_cell(stretch=stretch)
        apd_values.append(apd)
    return {
        'repol_hi': np.std(apd_values) / np.mean(apd_values),
        'apd_range': max(apd_values) - min(apd_values),
        'apd_std': np.std(apd_values),
    }

def run_threshold():
    """
    Fine-grained asymmetry sweep at fixed severity=1.0.
    Identify the asymmetry threshold where repolarization heterogeneity
    rises sharply — the geometric tipping point for arrhythmia substrate.
    
    Arrhythmia threshold reference:
    APD dispersion >10ms associated with increased VT inducibility
    in experimental DCM models (Laurita & Rosenbaum 2000, Circ Res).
    """
    
    # Fine sweep of asymmetry at full severity
    asymmetries = np.linspace(0.0, 1.0, 21)  # 0.05 steps
    
    results = []
    print('Threshold analysis: fine asymmetry sweep (severity=1.0)')
    print(f'{"Asymmetry":>12} {"APD range (ms)":>16} {"Repol HI":>12} {"S/L gradient":>14}')
    print('-' * 56)
    
    for asym in asymmetries:
        lv = dcm_asymmetric(severity=1.0, asymmetry=asym)
        metrics = compute_repol_heterogeneity(lv)
        sl_grad = lv.septal_lateral_gradient()
        results.append({
            'asymmetry': asym,
            'sl_gradient': sl_grad,
            **metrics
        })
        print(f'{asym:>12.2f} {metrics["apd_range"]:>16.2f} '
              f'{metrics["repol_hi"]:>12.4f} {sl_grad:>14.4f}')
    
    # Identify threshold — where APD range crosses 5ms
    # (conservative threshold; literature suggests >10ms is clearly arrhythmogenic)
    THRESHOLD_MS = 5.0
    threshold_asym = None
    for r in results:
        if r['apd_range'] >= THRESHOLD_MS:
            threshold_asym = r['asymmetry']
            threshold_sl = r['sl_gradient']
            break
    
    print(f'\nAPD dispersion threshold ({THRESHOLD_MS}ms) crossed at:')
    print(f'  Asymmetry index: {threshold_asym:.2f}')
    print(f'  Septal/lateral gradient: {threshold_sl:.2f}')
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Geometric Threshold for Arrhythmogenic Repolarization Substrate',
                 fontsize=12, fontweight='bold')
    
    x = [r['asymmetry'] for r in results]
    apd_range = [r['apd_range'] for r in results]
    sl_grad = [r['sl_gradient'] for r in results]
    repol_hi = [r['repol_hi'] for r in results]
    
    # Panel A: APD range vs asymmetry with threshold line
    ax = axes[0]
    ax.plot(x, apd_range, 'o-', color='crimson', linewidth=2, markersize=5)
    ax.axhline(THRESHOLD_MS, color='black', linestyle='--', linewidth=1.5,
               label=f'Arrhythmia threshold ({THRESHOLD_MS}ms)')
    ax.axhline(10.0, color='gray', linestyle=':', linewidth=1.5,
               label='High-risk threshold (10ms)')
    if threshold_asym is not None:
        ax.axvline(threshold_asym, color='orange', linestyle='--', linewidth=1.5,
                   label=f'Threshold asymmetry ({threshold_asym:.2f})')
    ax.set_xlabel('Remodeling Asymmetry Index')
    ax.set_ylabel('APD Dispersion (ms)')
    ax.set_title('A. APD Dispersion vs Asymmetry')
    ax.legend(fontsize=8)
    ax.fill_between(x, apd_range, THRESHOLD_MS,
                    where=[a >= THRESHOLD_MS for a in apd_range],
                    alpha=0.15, color='red', label='Arrhythmogenic zone')
    
    # Panel B: Septal/lateral gradient vs APD range
    ax = axes[1]
    sc = ax.scatter(sl_grad, apd_range, c=x, cmap='plasma', s=60, zorder=3)
    ax.axhline(THRESHOLD_MS, color='black', linestyle='--', linewidth=1.5,
               label=f'Threshold ({THRESHOLD_MS}ms)')
    plt.colorbar(sc, ax=ax, label='Asymmetry Index')
    
    # Fit
    z = np.polyfit(sl_grad, apd_range, 1)
    p = np.poly1d(z)
    xline = np.linspace(min(sl_grad), max(sl_grad), 100)
    ax.plot(xline, p(xline), 'k--', alpha=0.4)
    
    r_val = np.corrcoef(sl_grad, apd_range)[0,1]
    ax.text(0.05, 0.92, f'r = {r_val:.3f}', transform=ax.transAxes, fontsize=10)
    ax.set_xlabel('Septal/Lateral Stress Gradient')
    ax.set_ylabel('APD Dispersion (ms)')
    ax.set_title('B. S/L Gradient → APD Dispersion')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('figures/threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('Figure saved to figures/threshold_analysis.png')
    
    return results, threshold_asym

if __name__ == '__main__':
    results, threshold = run_threshold()