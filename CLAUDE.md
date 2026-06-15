# Project: Selective Prediction & Calibrated Abstention for Multi-Label ECG

## Goal
Q1/Q2 paper on calibration + conformal prediction + selective abstention 
for PTB-XL multi-label ECG classification. Baseline target: 
reproduce Strodthoff et al. xresnet1d101 macro AUROC ~0.925 on superdiagnostic.

## Hardware constraints
MacBook Pro M3 Max. PyTorch MPS backend. No CUDA.

## Code conventions
- One module per concern; see directory layout below
- All metrics defined in src/metrics.py with docstrings citing source papers
- Save logits to disk (ModelOutputs pickle) after every training run
- Never run on test fold during development — fold 9 is val, fold 10 is final test

## Directory layout


## What Claude Code should NOT do
- Do not modify data/ — that's downloaded datasets, read-only
- Do not modify outputs/*.pkl — trained model logits, expensive to regenerate
- Do not invent metric definitions — ask before adding new metrics
- Never use random splits — always use PTB-XL official strat_fold column
- Do not change the baseline xresnet1d101 architecture without explicit instruction
