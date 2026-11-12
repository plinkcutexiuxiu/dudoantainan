# SafeRoute · US Accidents

## Scope and source

The model classifies the dataset's four traffic-impact levels for a recorded
incident: Level 1 is the lowest reported impact and Level 4 the highest. These
labels do **not** mean injury/fatality classes, and the model does not predict
whether an accident will happen or estimate causal risk.

Source: [US Accidents (2016–2023), Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
The project uses the 500,000-row sample linked by the dataset author, downloaded
from the linked Google Drive file. The dataset covers 49 US states and contains
date/time, coordinates, weather, traffic-device flags, and severity. The source
page states CC BY-NC-SA 4.0 and restricts use to academic/non-commercial
research; cite the dataset and its referenced papers in submitted work.

Model inputs use start time/day, state, weather, sunrise/sunset, selected
environment readings, and traffic-infrastructure flags. Coordinates are used
only for the map, never as model inputs. `End_Time`, end coordinates, distance,
incident description, and identifiers are excluded because they can encode
post-start or post-outcome information.

## Algorithms and evaluation

- **Majority baseline:** `DummyClassifier(most_frequent)` provides a simple
  lower bound. The dataset is highly imbalanced, so baseline accuracy can look
  strong while minority-class recall is zero.
- **Decision Tree/CART:** handles nonlinear interactions in mixed tabular data
  without scaling. Gini selects splits; `max_depth`, `min_samples_leaf`, and
  `ccp_alpha` constrain complexity. Balanced class weights increase the
  influence of rare severity levels.
- **Random Forest:** averages trees fit on bootstrap samples with random
  feature subsets, generally reducing single-tree variance. The main controls
  are `n_estimators`, `max_features`, `min_samples_leaf`, and class weighting.

The script uses a fixed stratified 80/20 train/test split, then 5-fold
`StratifiedKFold` on training data only. Model choice uses training-fold mean
macro-F1; the test set is evaluated only after selection. Imputation and
one-hot encoding remain inside `Pipeline`/`ColumnTransformer`, so each fold
fits its own preprocessing. It reports mean/std for CV macro-F1 and balanced
accuracy, final test metrics, per-class scores, confusion matrix, ROC-AUC,
permutation importance, and error groups.

## Run

From PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements-ml.txt
py train_kaggle_accidents.py
```

On first run, the trainer retrieves the author-linked sample, extracts the
CSV, creates charts/map, and trains on a stratified 60,000-row sample. Use
`--sample-size 0` for all 500,000 rows or `--folds 3` for a faster iteration.
The full CSV is large; keep the original archive and extracted data in
`data/` rather than committing them to Git.

Start the local web demo after training:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8008
```

Open `http://127.0.0.1:8008/`. The demo account is local-browser-only; a public
deployment needs real authentication and a host for the Python API/model. The
Kaggle license is non-commercial, so do not deploy the dataset or model as a
commercial service without permission.

## Visual outputs

`artifacts/` contains the model comparison, per-class report, confusion matrix,
permutation importance, error examples, class balance, time/state analytics,
dataset profile, and `accident_map.html` with a heat layer and sampled points.
The dashboard serves the map and plots from the same API as predictions.
The `/api/health` check loads and validates the model, then keeps it cached
until the process restarts; the dashboard refreshes its API status every 30
seconds. A healthy response therefore means the prediction model is available,
not just that its file exists.

Interpret macro-F1 and per-level recall alongside accuracy. A large
train-CV macro-F1 gap suggests overfitting; error groups and feature importance
are descriptive, not causal. Validate by later time periods and unseen states
before considering generalization.

## Viva and practical exercise

  CART chooses a split that reduces weighted child impurity. `max_depth`,
  `min_samples_leaf`, class weights, and `ccp_alpha` affect tree complexity.
  and combines tree votes. This reduces variance but limits interpretability.
  Precision measures correctness among predictions of a class, recall measures
  how many actual records of that class are found.
  `train_kaggle_accidents.py`, run `py train_kaggle_accidents.py --sample-size
  10000`, and compare CV mean/std with final test macro-F1. Do not select
  parameters from the test score.

Always describe Level 1–4 as the source dataset's traffic-impact severity, not
injury severity or a probability that an accident will occur.

## Permanent public link

The repository includes a Docker image and `render.yaml` blueprint. To create
a stable URL, push this folder to a GitHub repository, then create a Render
Blueprint from that repository. The blueprint uses an always-on Starter web
service and sets `SAFEROUTE_PUBLIC_DEMO=1`, so opening the service root goes
straight to the dashboard. Render supplies a stable `*.onrender.com` address.
The always-on plan is paid; a free sleeping service does not meet a strict
24/7 requirement. The app is currently in a local folder without a Git remote,
so the GitHub repository and Render account still need to be provided/created
before it can be deployed. Do not publish `data/` or personal credentials.