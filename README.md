# DCM Arrhythmia Model

Computational model linking asymmetric left ventricular wall-stress heterogeneity to a stretch-activated arrhythmogenic substrate in dilated cardiomyopathy. The model couples a parametric prolate-ellipsoid LV geometry, regional biaxial (Laplace) wall stress, and the O'Hara-Rudy 2011 human ventricular action potential model with a stretch-activated channel (SAC) current engaged in diastole. The central result is that asymmetric remodeling produces a regional diastolic-depolarization and repolarization substrate that uniform dilation at the same ejection fraction does not, scaling with an echo-computable septal/lateral wall-thickness ratio.

Associated paper: "Asymmetric Wall-Stress Heterogeneity Defines a Stretch-Activated Arrhythmogenic Substrate Beyond Ejection Fraction in Dilated Cardiomyopathy: An In Silico Study" (Frontiers in Physiology, Cardiac Electrophysiology).

## Structure
- models/: LV geometry and single-cell electrophysiology (geometry.py, single_cell.py)
- experiments/: diastolic sweep, parameter sweep, threshold analysis, sensitivity analysis, validation
- data/: generated output files
- ohara_rudy_2011.mmt: O'Hara-Rudy 2011 model file for myokit (epicardial cell, cell.mode = 1)

## Requirements
Python 3.11, myokit (SUNDIALS/CVODE), numpy, matplotlib

## Usage
python3 experiments/diastolic_sweep.py      # primary readout: regional resting (diastolic) Vm substrate
python3 experiments/threshold_analysis.py   # substrate vs septal/lateral stress gradient
python3 experiments/parameter_sweep.py      # stress and substrate across the 25 geometries
python3 experiments/sensitivity_analysis.py # SAC conductance and reversal potential sensitivity
python3 experiments/validation.py           # single-cell SAC validation against the reported envelope