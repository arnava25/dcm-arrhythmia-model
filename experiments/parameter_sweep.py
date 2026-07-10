import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from models.geometry import LVGeometry, dcm_uniform, dcm_asymmetric, normal_lv
from models.single_cell import run_single_cell

def compute_repolarization_heterogeneity(lv_geometry):
    """
    For a given LV geometry:
    1. Compute regional stretch map
    2. Run single cell model for each region's stretch
    3. Return APD distribution and heterogeneity metrics
    """
    stretch_map = lv_geometry.stretch_map()
    
    apd_map = {}
    for region, stretch in stretch_map.items():
        apd = run_single_cell(stretch=stretch)
        apd_map[region] = apd
    
    apd_values = list(apd_map.values())
    
    return {
        'apd_map': apd_map,
        'apd_mean': np.mean(apd_values),
        'apd_std': np.std(apd_values),
        'apd_range': max(apd_values) - min(apd_values),
        'repol_heterogeneity': np.std(apd_values) / np.mean(apd_values),
        'stress_heterogeneity': lv_geometry.heterogeneity_index(),
        'septal_lateral_gradient': lv_geometry.septal_lateral_gradient(),
    }


def run_sweep():
    """
    Systematic sweep across DCM geometry parameter space.
    Severity: degree of dilation and thinning
    Asymmetry: degree of septal vs lateral differential remodeling
    """
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    asymmetries = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    results = []
    total = len(severities) * len(asymmetries)
    count = 0
    
    print(f'Running parameter sweep: {total} geometries')
    print(f'{"Severity":>10} {"Asymmetry":>10} {"APD range":>12} {"Repol HI":>10} {"Stress HI":>10} {"S/L grad":>10}')
    print('-' * 64)
    
    for severity in severities:
        for asymmetry in asymmetries:
            count += 1
            
            if severity == 0.0:
                lv = normal_lv()
            else:
                lv = dcm_asymmetric(severity=severity, asymmetry=asymmetry)
            
            metrics = compute_repolarization_heterogeneity(lv)
            
            result = {
                'severity': severity,
                'asymmetry': asymmetry,
                **metrics,
                # flatten apd_map for saving
                'apd_map': metrics['apd_map']
            }
            results.append(result)
            
            print(f'{severity:>10.2f} {asymmetry:>10.2f} '
                  f'{metrics["apd_range"]:>12.2f} '
                  f'{metrics["repol_heterogeneity"]:>10.4f} '
                  f'{metrics["stress_heterogeneity"]:>10.4f} '
                  f'{metrics["septal_lateral_gradient"]:>10.4f}')
    

    with open('data/sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved to data/sweep_results.json')
    return results


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    results = run_sweep()
    

    print('\n=== Key finding ===')
    normal = [r for r in results if r['severity'] == 0.0][0]
    uniform_dcm = [r for r in results if r['severity'] == 1.0 and r['asymmetry'] == 0.0][0]
    asymmetric_dcm = [r for r in results if r['severity'] == 1.0 and r['asymmetry'] == 1.0][0]
    
    print(f'Normal LV:          APD range = {normal["apd_range"]:.2f} ms, '
          f'repol HI = {normal["repol_heterogeneity"]:.4f}')
    print(f'Uniform DCM:        APD range = {uniform_dcm["apd_range"]:.2f} ms, '
          f'repol HI = {uniform_dcm["repol_heterogeneity"]:.4f}')
    print(f'Asymmetric DCM:     APD range = {asymmetric_dcm["apd_range"]:.2f} ms, '
          f'repol HI = {asymmetric_dcm["repol_heterogeneity"]:.4f}')