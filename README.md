# DCM Arrhythmia Model

Computational model linking LV wall stress heterogeneity to arrhythmogenic repolarization substrate in dilated cardiomyopathy.

Associated paper: Asymmetric Wall Stress Heterogeneity as a Determinant of Arrhythmogenic Repolarization Substrate in Dilated Cardiomyopathy: An In Silico Study

## Structure
- models/ — LV geometry and single cell electrophysiology
- experiments/ — parameter sweep, sensitivity analysis, threshold analysis, validation
- data/ — generated output files
- ohara_rudy_2011.mmt — O'Hara-Rudy 2011 model file for myokit

## Requirements
Python 3.11, myokit, numpy, matplotlib, sundials

## Usage
python3 experiments/parameter_sweep.py
python3 experiments/threshold_analysis.py
python3 experiments/sensitivity_analysis.py
python3 experiments/validation.py
