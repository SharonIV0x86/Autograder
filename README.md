# AutoGrader AI — ML-Based Code Evaluation Engine (Prototype)

This prototype extracts static features from submitted code (Python and basic C/C++ heuristics),
trains a small RandomForest classifier on synthetic labels, and exposes:

- CLI: analyze individual files
- Web UI: upload file, see features + PASS/FAIL prediction
- Training script: generate synthetic data and train model

## Quickstart

```bash
# 1. clone repo and cd into it
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# 2. train the synthetic model
python -m src.autograder.train    # will create src/autograder/model.joblib

# 3. run web UI
python -m src.autograder.webapp   # opens at http://127.0.0.1:8080

# 4. CLI usage
python -m src.autograder.cli analyze path/to/code.py
