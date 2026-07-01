# CDF-Task

This repository now contains a small Python project that compares common feature scaling methods: standardization, min-max scaling, and robust scaling.

## Files
- feature_scaling.py: implementation of the scaling methods and a comparison summary
- tests/test_feature_scaling.py: regression tests for the scaling functions
- requirements.txt: Python dependencies

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python feature_scaling.py
```