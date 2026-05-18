import sys
sys.path.insert(0, '/Users/aru/Desktop/dcm')

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load results
with open('data/sweep_results.json') as f:
    results = json.load(f)

# Exclude severity=0 (normal, asymmetry has no effect there)
dcm = [r for r in results if r['severity'] > 0.0]
normal = [r for r in results if r['severity'] == 0.0][0]

severities = sorted(set(r['severity'] for r in dcm))
asymmetries = sorted(set(r['asymmetry'] for r in dcm))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Wall Stress Heterogeneity and Repolarization Substrate in DCM', 
             fontsize=13, fontweight='bold')

colors = cm.viridis(np.linspace(0.2, 0.9, len(severities)))

# Panel 1: Stress heterogeneity vs asymmetry, by severity
ax = axes[0]
for i, sev in enumerate(severities):
    subset = [r for r in dcm if r['severity'] == sev]
    subset.sort(key=lambda r: r['asymmetry'])
    x = [r['asymmetry'] for r in subset]
    y = [r['stress_heterogeneity'] for r in subset]
    ax.plot(x, y, 'o-', color=colors[i], label=f'Severity {sev}')
ax.axhline(normal['stress_heterogeneity'], color='gray', linestyle='--', label='Normal')
ax.set_xlabel('Remodeling Asymmetry')
ax.set_ylabel('Stress Heterogeneity Index (CV)')
ax.set_title('A. Wall Stress Heterogeneity')
ax.legend(fontsize=8)

# Panel 2: Repolarization heterogeneity vs asymmetry, by severity
ax = axes[1]
for i, sev in enumerate(severities):
    subset = [r for r in dcm if r['severity'] == sev]
    subset.sort(key=lambda r: r['asymmetry'])
    x = [r['asymmetry'] for r in subset]
    y = [r['repol_heterogeneity'] for r in subset]
    ax.plot(x, y, 'o-', color=colors[i], label=f'Severity {sev}')
ax.axhline(normal['repol_heterogeneity'], color='gray', linestyle='--', label='Normal')
ax.set_xlabel('Remodeling Asymmetry')
ax.set_ylabel('Repolarization Heterogeneity Index (CV of APD90)')
ax.set_title('B. Repolarization Heterogeneity')
ax.legend(fontsize=8)

# Panel 3: Stress HI vs Repol HI scatter — the key relationship
ax = axes[2]
x_all = [r['stress_heterogeneity'] for r in dcm]
y_all = [r['repol_heterogeneity'] for r in dcm]
sev_all = [r['severity'] for r in dcm]
sc = ax.scatter(x_all, y_all, c=sev_all, cmap='viridis', s=60, zorder=3)
ax.scatter(normal['stress_heterogeneity'], normal['repol_heterogeneity'],
           color='red', marker='*', s=200, zorder=4, label='Normal LV')

# Fit a line
z = np.polyfit(x_all, y_all, 1)
p = np.poly1d(z)
xline = np.linspace(min(x_all), max(x_all), 100)
ax.plot(xline, p(xline), 'k--', alpha=0.5, label='Linear fit')

plt.colorbar(sc, ax=ax, label='DCM Severity')
ax.set_xlabel('Stress Heterogeneity Index')
ax.set_ylabel('Repolarization Heterogeneity Index')
ax.set_title('C. Stress → Repolarization Coupling')
ax.legend(fontsize=8)

# Compute R^2
correlation = np.corrcoef(x_all, y_all)[0,1]
ax.text(0.05, 0.92, f'r = {correlation:.3f}', transform=ax.transAxes, fontsize=10)

plt.tight_layout()
plt.savefig('figures/main_figure.png', dpi=300, bbox_inches='tight')
plt.show()
print('Figure saved to figures/main_figure.png')