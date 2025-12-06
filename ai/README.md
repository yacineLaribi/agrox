# Agrox AI — Final Training

This repository contains scripts and data used to train a model for the Agrox project.

Important artifacts
- `final-training-using-different-params.py` — The final training script that performs model training while sweeping/trying different parameters.
- `.pkl` — The trained model is saved as a pickled (`.pkl`) file when training completes. This file is the final model artifact you can load for inference or evaluation.
- `report.md` — A detailed explanation of the code, training decisions, experiments and results. See this file for deeper context and methodology.

Accuracy
- Observed accuracy for the final trained models ranges roughly between 82% and 90% depending on how training pairs are generated and which parameter combinations are used during the final training run.

Quick reproduction notes

1. Prerequisites
   - Python 3.8+ (use the environment you normally run Python in).
   - Install any project dependencies. If a `requirements.txt` is not present, inspect the training script for imports and install the required packages (common packages for ML projects: scikit-learn, pandas, numpy, joblib/pickle, etc.).

2. Data
   - Input data is stored in the `dataset/` folder (e.g. `family_data.csv`, `genus_data.csv`). The cleaning pipeline lives under `cleaning/` and includes scripts to prepare `data.csv`.

3. Run training (basic)
   - The simplest way to run the final training (adjust paths/args as needed):

```bash
python3 final-training-using-different-params.py
```

   - The script will produce a `.pkl` file containing the trained model. Check the script for the exact output filename and any CLI arguments (for example, output path, random seed, or parameter grid choices).

Notes and assumptions
- This README intentionally keeps commands generic. If you want, I can update this README with the exact command-line arguments, the exact `.pkl` filename and a `requirements.txt` based on the project imports.
- Accuracy depends strongly on how pairs are generated for training, the parameter choices, and random seeds. Use `report.md` for a deeper breakdown of experiments that produced the reported 82%–90% range.

Where to look next
- `report.md` — read this for explanations of the preprocessing, modeling choices, and evaluation methodology.
- `cleaning/` — the data cleaning pipeline and any helper scripts.
- `final-training-using-different-params.py` and `training-final.py` — review these files to find exact CLI flags and output filenames.

If you want, I can:
- Insert exact run commands with CLI flags into this README (I will read the training script to extract them).
- Create a `requirements.txt` automatically by scanning imports.
- Add a short section that shows how to load the `.pkl` model for inference.

— End of README —
