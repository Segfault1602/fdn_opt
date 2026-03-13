import subprocess
import math
from multiprocessing import Pool
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import optuna

EXE_NAME = "tuning.exe"
path_to_exe = (
    Path(__file__).parent.parent.parent
    / "build"
    / "ninja"
    / "src"
    / "optimization"
    / "Release"
    / EXE_NAME
)

OPTIM_TYPE = "CMAES"


def objective(trial):
    args = [str(path_to_exe), OPTIM_TYPE]
    if OPTIM_TYPE == "ADAM":
        step_size = trial.suggest_float("step_size", 0.01, 1.0)
        lr_decay = trial.suggest_float("lr_decay", 0.1, 1.0)
        decay_step_size = trial.suggest_int("decay_step_size", 1, 100)
        epoch_restarts = trial.suggest_int("epoch_restarts", 50, 500)
        max_restarts = trial.suggest_int("max_restarts", 0, 10)
        tolerance = trial.suggest_int("tolerance_exp", -7, -2)
        tolerance = 10**tolerance
        args += [
            str(step_size),
            str(lr_decay),
            str(decay_step_size),
            str(epoch_restarts),
            str(max_restarts),
            str(tolerance),
        ]
    elif OPTIM_TYPE == "SPSA":
        alpha = trial.suggest_float("alpha", 0.001, 1.0)
        gamma = trial.suggest_float("gamma", 0.001, 1.0)
        step_size = trial.suggest_float("step_size", 0.1, 2.0)
        eval_step_size = trial.suggest_float("eval_step_size", 0.0001, 2.0)
        max_iterations = trial.suggest_int("max_iterations", 2, 10)
        max_iterations = 10**max_iterations
        tolerance = trial.suggest_int("tolerance_exp", -7, -2)
        tolerance = 10**tolerance

        args += [
            str(alpha),
            str(gamma),
            str(step_size),
            str(eval_step_size),
            str(max_iterations),
            str(tolerance),
        ]
    elif OPTIM_TYPE == "CMAES":
        population_size = trial.suggest_int("population_size", 0, 1000)
        max_iterations = trial.suggest_int("max_iterations", 1, 10)
        max_iterations = 10**max_iterations
        tolerance = trial.suggest_int("tolerance_exp", -7, -2)
        tolerance = 10**tolerance
        step_size = trial.suggest_float("step_size", 0.1, 2.0)

        args += [
            str(population_size),
            str(max_iterations),
            str(tolerance),
            str(step_size),
        ]
    # Run the executable
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except subprocess.TimeoutExpired as e:
        print("Subprocess timed out:")
        raise optuna.TrialPruned() from e
    except subprocess.CalledProcessError as e:
        print("Subprocess failed:")
        print(e.stdout)
        print(e.stderr)
        raise optuna.TrialPruned() from e

    best_loss = 9999.0
    elapsed_time = 0.0

    # Find string "Best loss: ####" in the output
    for line in result.stdout.splitlines():
        if "Best loss:" in line:
            best_loss = float(line.strip().split()[-1])
        if "Final time:" in line:
            elapsed_time = float(line.strip().split()[-2])

    # print(f"Best loss from tuning: {best_loss}")
    # print(f"Elapsed time from tuning: {elapsed_time} seconds")
    # return best_loss, elapsed_time
    trial.set_user_attr("elapsed_time", elapsed_time)
    return best_loss, elapsed_time


def run_optimization(_):
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        sampler=sampler,
        storage="sqlite:///db.sqlite3",
        study_name=f"{OPTIM_TYPE} - Spectrum/time (TPESampler)",
        load_if_exists=True,
        directions=["minimize", "minimize"],
    )
    study.set_metric_names(["loss", "time"])
    study.optimize(objective, n_trials=100)


if __name__ == "__main__":

    # Check if the executable exists
    if not path_to_exe.is_file():
        print(f"Executable not found at {path_to_exe}")
        exit(1)

    N_PROCESSES = 16
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(run_optimization, range(N_PROCESSES))
