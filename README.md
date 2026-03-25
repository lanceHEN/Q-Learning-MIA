# Q Learning MIA

This project focuses on MIAs against tabular Q learning algorithms. Essentially,
they leverage the idea that commonly visited state-action pairs should have
lower Bellman residuals. Since trajectories used in training will typically have
more of these pairs, they should yield a lower residual than average. This distinguishment
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
├──component_config.py 
├──experiment_config.py 
├──__init__.py
src/
├── experimentation/
│   ├── experiment.py
│   └── __init__.py
└── model/
    ├── data_oracle.py
    ├── mia.py
    ├── trainer_oracle.py
    └── __init__.py
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
it is a training trajectory or not via a likelihood ratio test (LRT).