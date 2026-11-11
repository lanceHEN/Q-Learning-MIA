"""
Runs one of the paper's experiment sweeps and writes its results to
data/tables/hyperparam_results_<name>.csv and data/tables/summary_<name>.csv
(one raw results file per run goes to data/raw_results/).

    python run_experiment.py --env sepsis --method baseline --member tabular
    python run_experiment.py --env glucose --method baseline --member cql --no-grid
    python run_experiment.py --env sepsis --method ensemble --member cql
    python run_experiment.py --env glucose --method dpapvi
    python run_experiment.py --env sepsis --method dpsgd

Methods:
    baseline  non-private attack target: one learner (k = 1), no noise, unclipped Q values
    ensemble  subsample-and-aggregate ensemble over K_VALUES and EPSILONS, Q values clipped to [0, 1]
    dpapvi    DP-APVI over EPSILONS (tabular)
    dpsgd     DP-SGD-trained CQL over EPSILONS (deep)

Every sweep uses the dataset sizes N_TRAJECTORIES and 50 repeats (see src/experimentation/setups.py).
"""
import argparse
from functools import partial

import numpy as np

from src.experimentation.runner import run_sweep, summarize
from src.experimentation import setups


def experiment_name(args) -> str:
    if args.method in ("dpapvi", "dpsgd"):
        return f"{args.env}-{args.method}"
    name = f"{args.env}-{args.member}-ensemble"
    if args.method == "baseline":
        name += "-baseline" + ("-nogrid" if args.no_grid else "")
    return name


def sweep_settings(args, name: str) -> dict:
    """The build function and grid of settings for run_sweep."""
    if args.method in ("baseline", "ensemble"):
        baseline = args.method == "baseline"
        build = partial(
            setups.build_ensemble_config,
            env_name=args.env,
            member=args.member,
            name=name,
            # Clipping is part of the mitigation: the baseline releases the learner's raw Q values
            q_bounds=(float("-inf"), float("inf")) if baseline else (0.0, 1.0),
            release_on_grid=not args.no_grid,
        )
        return dict(
            build_config=build,
            n_workers=args.workers or 6,
            trainer_oracle_epochs_list=[setups.EPOCHS[args.member]],
            ensemble_k_list=[1] if baseline else setups.K_VALUES,
            ensemble_epsilon_list=[None] if baseline else setups.EPSILONS,
            share_training_across_epsilons=True,  # train once per (N, k, repeat), release at every epsilon
        )
    if args.method == "dpapvi":
        return dict(
            build_config=partial(setups.build_dpapvi_config, env_name=args.env, name=name),
            n_workers=args.workers or 6,
            trainer_oracle_epochs_list=[1],  # unused by DP-APVI
            ensemble_k_list=[1],  # unused by DP-APVI
            ensemble_epsilon_list=setups.EPSILONS,
        )
    return dict(
        build_config=partial(setups.build_dpsgd_config, env_name=args.env, name=name),
        n_workers=args.workers or 4,
        trainer_oracle_epochs_list=[setups.EPOCHS["dpsgd"]],
        ensemble_k_list=[1],  # unused by DP-SGD
        ensemble_epsilon_list=setups.EPSILONS,
    )


def report_data_oracle(env_name: str, data_oracle):
    """Prints the behavior policy's mean return and how often it takes its greedy action."""
    eval_trajs = data_oracle.generate_trajectories(500, setups.T_MAX[env_name])
    mean_reward = np.mean([sum(r for _, _, r, _ in traj) for traj in eval_trajs])
    print(f"Data oracle mean reward: {mean_reward:.4f}")
    ps = []
    for traj in eval_trajs:
        for state, *_ in traj:
            q_vals = data_oracle.q_vector(state)
            probs = np.exp((q_vals - np.max(q_vals)) / data_oracle.config.tau)
            ps.append(np.max(probs / probs.sum()))
    print(f"Expected greedy selection %: {np.mean(ps):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, choices=setups.ENVS)
    parser.add_argument("--method", required=True, choices=["baseline", "ensemble", "dpapvi", "dpsgd"])
    parser.add_argument("--member", choices=["tabular", "cql"], help="learner for baseline and ensemble")
    parser.add_argument("--no-grid", action="store_true",
                        help="baseline CQL only: query the network at each exact observation instead of the grid")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--workers", type=int, help="parallel worker processes (default 6; 4 for dpsgd)")
    parser.add_argument("--epsilons", nargs="+",
                        help="run only these privacy budgets instead of the method's full grid; 'none' = no noise")
    parser.add_argument("--tag", help="suffix for the results file names, e.g. to rerun some settings without "
                                      "overwriting the full table (raw per-run files keep their usual names)")
    args = parser.parse_args()

    if args.method in ("baseline", "ensemble") and args.member is None:
        parser.error(f"--method {args.method} needs --member")
    if args.method in ("dpapvi", "dpsgd") and args.member is not None:
        parser.error(f"--member does not apply to --method {args.method}")
    if args.no_grid and not (args.method == "baseline" and args.member == "cql"):
        parser.error("--no-grid applies only to --method baseline --member cql (the noise needs a finite table)")

    report_data_oracle(args.env, setups.load_or_train_data_oracle(args.env))
    if args.method == "dpapvi":
        setups.dpapvi_reward_table(args.env, setups.make_tabular_env(args.env))  # build it once, before the workers start

    name = experiment_name(args)
    table_name = name.replace("-", "_") + (f"_{args.tag}" if args.tag else "")
    csv_path = f"data/tables/hyperparam_results_{table_name}.csv"
    summary_path = f"data/tables/summary_{table_name}.csv"

    settings = sweep_settings(args, name)
    if args.epsilons:
        settings["ensemble_epsilon_list"] = [None if e.lower() == "none" else float(e) for e in args.epsilons]
    table = run_sweep(
        n_repeats=args.repeats,
        n_trajectories_list=setups.N_TRAJECTORIES,
        csv_path=csv_path,
        **settings,
    )
    table.to_csv(csv_path)
    print(table)

    summary = summarize(table)
    summary.to_csv(summary_path)
    print("\nSummary (mean ± std):")
    print(summary)


if __name__ == "__main__":
    main()
