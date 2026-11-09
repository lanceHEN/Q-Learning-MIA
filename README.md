# Q Learning MIA

This project focuses on MIAs against tabular and deep Q learning algorithms. Essentially,
they leverage the idea that commonly visited state-action pairs should have
lower squared Bellman residuals. Since trajectories used in training will typically have
more of these pairs, they should yield a lower squared residual on average. This distinguishment
can be used to assess whether a given trajectory was used in training.

## Repository Overview

### Installation
1. Clone the repository:
```bash
git clone https://github.com/lanceHEN/Q-Learning-MIA.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## File Structure
```text
config/
├── component_config.py
├── experiment_config.py
└── __init__.py
src/
├── experimentation/
│   ├── runner.py
│   ├── setups.py
│   ├── roc_analysis.py
│   └── __init__.py
└── model/
    ├── cql_dqn.py
    ├── data_oracle.py
    ├── generic_model.py
    ├── mia.py
    ├── trainer_oracle.py
    ├── wrappers.py
    └── __init__.py
run_experiment.py
.gitignore
README.md
requirements.txt
```

### config/
Stores helpful dataclasses to centralize config info. `component_config.py`
stores config dataclasses for model components, while `experiment_config.py`
stores config dataclasses for experiments.

### src/
Contains the full MIA architecture.

Includes the Data Oracle in `data_oracle.py`, which collects trajectories to either be held out
or used to train the Trainer Oracle in `trainer_oracle.py`. Finally, `mia.py`
contains the attack classifier that will, given a trajectory, determine whether
it is a training trajectory or not via $k$-means clustering ($k=2$).

### src/experimentation/
`runner.py` runs one attack (`ExperimentRunner`) and sweeps over settings in
parallel (`run_sweep`). `setups.py` holds each environment's settings and builds
the configs for the ensemble, DP-APVI, and DP-SGD experiments. `roc_analysis.py`
computes TPRs at low FPRs and ROC curves from the raw per-run results.

## Running Experiments
Run from the repository root. Each command runs one sweep (50 repeats per setting)
and writes `data/tables/hyperparam_results_<name>.csv` and `data/tables/summary_<name>.csv`:
```bash
python run_experiment.py --env sepsis --method baseline --member tabular
python run_experiment.py --env glucose --method baseline --member cql --no-grid
python run_experiment.py --env sepsis --method ensemble --member cql
python run_experiment.py --env glucose --method dpapvi
python run_experiment.py --env sepsis --method dpsgd
```
See `python run_experiment.py --help` for all options.
