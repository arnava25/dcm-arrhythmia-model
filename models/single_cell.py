import myokit
import numpy as np

import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ohara_rudy_2011.mmt')
BEAT_DURATION_MS = 1000          # 1 Hz pacing

# Pace a FIXED, identical number of beats for every cell, then measure the
# last beat. O'Hara-Rudy drifts slowly for hundreds of beats; pacing every
# cell the same number of beats makes that residual drift common to all
# regions so it cancels in the region-to-region APD dispersion. (The
# previous per-cell convergence stop let different stretches halt at
# different beat counts, injecting differential drift larger than the
# signal.) Verify drift is negligible at PACE_BEATS with return_diagnostics.
PACE_BEATS = 600

# SAC parameters. The ohmic (linear) form I_SAC = g(V - E_SAC) is the
# standard SAC model (Peyronnet, Nerbonne & Kohl, Circ Res 2016).
# Ventricular-myocyte SAC parameters: Kamkin, Kiseleva & Isenberg,
# Cardiovasc Res 2000;48:409-420 (linear I-V, reverses near 0 mV,
# amplitude rises with hypertrophy); primary single-channel identification
# Craelius, Chen & El-Sherif, Biosci Rep 1988;8:407-414 (~120 pS
# non-selective cation, reversal ~0 mV).
#
# E_SAC depends on the channel mix: the non-selective cation SAC reverses
# near 0 to -10 mV, the K-selective SAC near -90 mV, so whole-cell E_SAC
# spans roughly -90 to -10 mV. With E_SAC = -10 mV the ohmic form below
# produces APD90 SHORTENING with stretch, as in K-SAC / IK_SAC-dominant
# preparations. Direction is genuinely preparation-dependent (non-selective
# dominance can prolong APD90); the arrhythmogenic quantity is the regional
# SPREAD of repolarization, not its sign.
E_SAC = -10.0
G_SAC_MAX = 0.003                # nS/pF

# Conductance reaches its maximum at 20% stretch (lambda = 1.2).
SAC_MAX_STRETCH = 1.2


def sac_conductance(stretch, g_max=G_SAC_MAX):
    """Linear stretch-activated conductance, zero below lambda = 1.0 and
    capped at g_max at lambda = 1.2 (and above). The previous version had
    no ceiling, so lambda up to 1.3 gave 1.5 * g_max, contradicting
    'maximum conductance at 20% stretch'."""
    if stretch <= 1.0:
        return 0.0
    frac = (stretch - 1.0) / (SAC_MAX_STRETCH - 1.0)
    frac = min(frac, 1.0)
    return g_max * frac


def apd90_from_trace(times, voltages):
    """APD90 (ms) as the duration the membrane spends above the 90%
    repolarisation voltage in a single beat.

        V90 = Vpeak - 0.9 * (Vpeak - Vrest)

    Crossings are linearly interpolated. Returns 0.0 if no action
    potential is present. This replaces the previous call
    d.apd(threshold=0.9), where myokit interprets 0.9 as an absolute
    0.9 mV threshold, so the old code measured time above 0.9 mV rather
    than APD90.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(voltages, dtype=float)

    v_rest = float(v.min())
    v_peak = float(v.max())
    if (v_peak - v_rest) < 20.0:          # no real upstroke
        return 0.0

    v90 = v_peak - 0.9 * (v_peak - v_rest)
    peak_idx = int(np.argmax(v))

    # Upward crossing of V90 (before the peak)
    t_up = None
    for i in range(peak_idx):
        if v[i] < v90 <= v[i + 1]:
            t_up = t[i] + (v90 - v[i]) * (t[i + 1] - t[i]) / (v[i + 1] - v[i])
            break
    if t_up is None:
        t_up = t[0]

    # Downward crossing of V90 (after the peak)
    t_down = None
    for i in range(peak_idx, len(v) - 1):
        if v[i] >= v90 > v[i + 1]:
            t_down = t[i] + (v[i] - v90) * (t[i + 1] - t[i]) / (v[i] - v[i + 1])
            break
    if t_down is None:
        t_down = t[-1]

    return float(t_down - t_up)


def resting_vm_from_trace(voltages):
    """Maximum diastolic potential (most negative Vm), i.e. the resting
    membrane potential, in mV. With an inward SAC current at rest this
    becomes less negative (diastolic depolarisation)."""
    return float(np.asarray(voltages, dtype=float).min())


def end_diastolic_vm_from_trace(voltages):
    """Vm at the end of the diastolic interval (last sample before the next
    stimulus), in mV."""
    return float(np.asarray(voltages, dtype=float)[-1])


def _run_metrics(stretch, g_sac_max, beats):
    """Core simulation. Returns dict with APD90, resting Vm, end-diastolic
    Vm (all from the final beat), and the last-to-previous beat APD drift."""
    m, p, _ = myokit.load(MODEL_PATH)
    g_sac = sac_conductance(stretch, g_sac_max)

    V = m.get('membrane.V')
    V.set_rhs(
        '-(membrane.i_ion + stimulus.i_stim + %r * (membrane.V - (%r)))'
        % (g_sac, E_SAC)
    )

    s = myokit.Simulation(m, p)
    time_var = m.time().qname()

    if beats > 2:
        s.run((beats - 2) * BEAT_DURATION_MS, log=myokit.LOG_NONE)
    d_prev = s.run(BEAT_DURATION_MS, log=[time_var, 'membrane.V'])
    d_last = s.run(BEAT_DURATION_MS, log=[time_var, 'membrane.V'])

    apd_prev = apd90_from_trace(d_prev[time_var], d_prev['membrane.V'])
    apd = apd90_from_trace(d_last[time_var], d_last['membrane.V'])
    return {
        'apd90': apd,
        'v_rest': resting_vm_from_trace(d_last['membrane.V']),
        'v_ed': end_diastolic_vm_from_trace(d_last['membrane.V']),
        'beats': beats,
        'drift_ms': abs(apd - apd_prev),
    }


def run_single_cell(stretch=1.0, g_sac_max=G_SAC_MAX, beats=PACE_BEATS,
                    return_diagnostics=False):
    """APD90 (ms) of the final beat after fixed-beat pacing. Backward
    compatible: returns a float unless return_diagnostics is set."""
    m = _run_metrics(stretch, g_sac_max, beats)
    if return_diagnostics:
        return {'apd90': m['apd90'], 'beats': m['beats'], 'drift_ms': m['drift_ms']}
    return m['apd90']


def run_single_cell_metrics(stretch=1.0, g_sac_max=G_SAC_MAX, beats=PACE_BEATS):
    """Full metrics dict: apd90, v_rest (resting/diastolic Vm, mV), v_ed
    (end-diastolic Vm, mV), beats, drift_ms."""
    return _run_metrics(stretch, g_sac_max, beats)


if __name__ == '__main__':
    print('SAC effect on APD90 - fixed-beat stretch sweep')
    print('%10s %12s %12s %8s %10s' %
          ('Stretch', 'APD90 (ms)', 'dAPD (ms)', 'beats', 'drift(ms)'))
    print('-' * 56)

    base = run_single_cell(stretch=1.0, return_diagnostics=True)
    baseline = base['apd90']
    print('%10.2f %12.1f %12.1f %8d %10.4f' %
          (1.0, baseline, 0.0, base['beats'], base['drift_ms']))

    for stretch in [1.05, 1.10, 1.15, 1.20]:
        r = run_single_cell(stretch=stretch, return_diagnostics=True)
        print('%10.2f %12.1f %12.1f %8d %10.4f' %
              (stretch, r['apd90'], r['apd90'] - baseline,
               r['beats'], r['drift_ms']))

    print()
    print('drift(ms) is the last-to-previous beat APD90 change; it must be')
    print('far below the inter-region dispersion for the sweep to be trustworthy.')
    print('Report baseline APD90 and the O\'Hara-Rudy cell type in Methods.')