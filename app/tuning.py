import subprocess
import signal
from multiprocessing import Pool
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import optuna
import optunahub
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

EXE_NAME = "optim_tool.exe"
path_to_exe = (
    Path(__file__).parent.parent / "build" / "ninja" / "app" / "Release" / EXE_NAME
)

OPTIM_TYPES = [
    "Simulated Annealing",
    "Differential Evolution",
    "CNE",
    "PSO",
    "SPSA",
    "CMA-ES",
    "Adam",
    "L-BFGS",
    "Gradient Descent",
]

OPTIM_TYPES_WITH_GRADIENT = ["Adam", "L-BFGS", "Gradient Descent"]
COLORLESS_ONLY = True
FDN_SIZE = 8
DEBUG = False


def run_optimization(args, optim_type, timeout=500):
    # Run the executable
    if DEBUG:
        print("Running command:", " ".join(args))

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
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

    best_loss_colorless = 9999.0
    best_loss_spectrum = 9999.0
    elapsed_time = 0.0
    total_evaluations_colorless = 0
    total_evaluations_spectrum = 0

    log_string = ""

    # Find string "Final loss: ####", "Elapsed time: #### s" and "Total evaluation: ###" in the output
    for line in result.stdout.splitlines():
        if "[Colorless] Final loss:" in line:
            best_loss_colorless = float(line.strip().split()[-1])
        if "[Spectrum] Final loss:" in line:
            best_loss_spectrum = float(line.strip().split()[-1])
        if "[Colorless] Elapsed time:" in line:
            elapsed_time = float(line.strip().split()[-2])
        if "[Spectrum] Elapsed time:" in line:
            elapsed_time = float(line.strip().split()[-2])
        if "[Colorless] Total evaluations:" in line:
            total_evaluations_colorless = int(line.strip().split()[-1])
        if "[Spectrum] Total evaluations:" in line:
            total_evaluations_spectrum = int(line.strip().split()[-1])
        if f"Starting {optim_type} optimization" in line:
            log_string = line

    return {
        "best_loss_colorless": best_loss_colorless,
        "best_loss_spectrum": best_loss_spectrum,
        "elapsed_time": elapsed_time,
        "total_evaluations_colorless": total_evaluations_colorless,
        "total_evaluations_spectrum": total_evaluations_spectrum,
        "log_string": log_string,
    }


def objective(optim_type, trial):
    args = [str(path_to_exe)]

    # Common parameters
    args += [
        f"--num_channels={FDN_SIZE}",
        "--spectral_flatness_weight=1",
        "--sparsity_weight=0.5",
        "--edc_weight=0.1",
        "--mel_edr_weight=1",
        "--save_output=false",
        "--verbose=false",
        "--ir=../rirs/py_rirs/rir_dining_room.wav",
        "--randomize_initial_params=false",
    ]

    if COLORLESS_ONLY:
        args += ["-c"]
    else:
        args += ["--spectrum_only"]

    timeout = 500

    if optim_type == "Adam":
        step_size = trial.suggest_float("step_size", 0.01, 1.0, step=0.001)
        beta1 = trial.suggest_float("beta1", 0.8, 0.999, step=0.001)
        beta2 = trial.suggest_float("beta2", 0.8, 0.999, step=0.001)
        gradient_delta = trial.suggest_int("gradient_delta_exp", -6, -1)
        gradient_delta = 10**gradient_delta
        # tolerance = trial.suggest_int("tolerance_exp", -5, -2)
        # tolerance = 10**tolerance
        tolerance = 1e-5
        args += [
            "Adam",
            f"--step_size={step_size}",
            f"--beta1={beta1}",
            f"--beta2={beta2}",
            f"--tolerance={tolerance}",
            f"--gradient_delta={gradient_delta}",
        ]
    elif optim_type == "L-BFGS":
        num_basis = trial.suggest_int("num_basis", 3, 500)
        max_iterations = trial.suggest_int("max_iterations", 1e3, 1e7)
        # max_iterations = 10**max_iterations
        wolfe = trial.suggest_float("wolfe", 0.8, 0.999, step=0.001)
        # min_gradient_norm = trial.suggest_float("min_gradient_norm_exp", -7, -1)
        # min_gradient_norm = 10**min_gradient_norm
        min_gradient_norm = 1e-6
        # factor = trial.suggest_float("factor_exp", -16, -10)
        # factor = 10**factor
        factor = 1e-15
        max_line_search = trial.suggest_int("max_line_search", 10, 100)
        # min_step = trial.suggest_float("min_step", -20, -1)
        # min_step = 10**min_step
        min_step = 1e-20
        # max_step = trial.suggest_float("max_step", 1, 20)
        # max_step = 10**max_step
        max_step = 1e20
        gradient_delta = trial.suggest_int("gradient_delta_exp", -6, -1)
        gradient_delta = 10**gradient_delta

        args += [
            "L-BFGS",
            f"--num_basis={num_basis}",
            f"--max_iterations={max_iterations}",
            f"--wolfe={wolfe}",
            f"--min_gradient_norm={min_gradient_norm}",
            f"--factor={factor}",
            f"--max_line_search={max_line_search}",
            f"--min_step={min_step}",
            f"--max_step={max_step}",
            f"--gradient_delta={gradient_delta}",
        ]
    elif optim_type == "Gradient Descent":
        step_size = trial.suggest_float("step_size", 0.01, 2.0, step=0.001)
        max_iterations = trial.suggest_int("max_iterations", 2, 10)
        max_iterations = 10**max_iterations
        tolerance = trial.suggest_int("tolerance_exp", -5, -3)
        kappa = trial.suggest_float("kappa", 0.01, 0.999, step=0.001)
        phi = trial.suggest_float("phi", 0.01, 0.999, step=0.001)
        momentum = trial.suggest_float("momentum", 0.01, 0.999, step=0.001)
        min_gain = trial.suggest_int("min_gain", -10, -2)
        min_gain = 10**min_gain
        gradient_delta = trial.suggest_int("gradient_delta_exp", -6, -1)
        gradient_delta = 10**gradient_delta

        args += [
            "GradientDescent",
            f"--step_size={step_size}",
            f"--max_iterations={max_iterations}",
            f"--tolerance={10**tolerance}",
            f"--kappa={kappa}",
            f"--phi={phi}",
            f"--momentum={momentum}",
            f"--min_gain={min_gain}",
            f"--gradient_delta={gradient_delta}",
        ]
    elif optim_type == "SPSA":
        alpha = trial.suggest_float("alpha", 0.001, 1.0)
        gamma = trial.suggest_float("gamma", 0.001, 1.0)
        step_size = trial.suggest_float("step_size", 0.1, 2.0, step=0.001)
        eval_step_size = trial.suggest_float("eval_step_size", 0.0001, 2.0)
        max_iterations = trial.suggest_int("max_iterations", 2, 10)
        max_iterations = 10**max_iterations
        # tolerance = trial.suggest_int("tolerance_exp", -5, -2)
        # tolerance = 10**tolerance
        tolerance = 1e-5

        args += [
            "SPSA",
            f"--alpha={alpha}",
            f"--gamma={gamma}",
            f"--step_size={step_size}",
            f"--evaluation_step_size={eval_step_size}",
            f"--max_iterations={max_iterations}",
            f"--tolerance={tolerance}",
        ]
    elif optim_type == "Simulated Annealing":
        max_iterations = trial.suggest_int("max_iterations", 3, 7)
        max_iterations = 10**max_iterations
        initial_temp = trial.suggest_int("initial_temperature", 5, 10000)
        init_moves = trial.suggest_int("init_moves", 10, 10000)
        move_ctrl_sweep = trial.suggest_int("move_ctrl_sweep", 1, 500)
        max_tolerance_sweep = trial.suggest_int("max_tolerance_sweep", 1, 10)
        max_move_coeff = trial.suggest_float("max_move_coeff", 0.1, 40.0, step=0.001)
        init_move_coeff = trial.suggest_float("init_move_coeff", 0.1, 2.0, step=0.001)
        gain = trial.suggest_float("gain", 0.1, 2.0, step=0.001)
        # tolerance_exp = trial.suggest_int("tolerance_exp", -5, -2)
        # tolerance = 10**tolerance_exp
        tolerance = 1e-5

        args += [
            "SimulatedAnnealing",
            f"--max_iterations={max_iterations}",
            f"--initial_temperature={initial_temp}",
            f"--init_moves={init_moves}",
            f"--move_ctrl_sweep={move_ctrl_sweep}",
            f"--max_tolerance_sweep={max_tolerance_sweep}",
            f"--max_move_coeff={max_move_coeff}",
            f"--init_move_coeff={init_move_coeff}",
            f"--gain={gain}",
            f"--tolerance={tolerance}",
        ]
    elif optim_type == "CNE":
        population_size = trial.suggest_int("population_size", 100, 10000, step=10)
        max_generations = trial.suggest_int("max_generations", 10, 10000, step=10)
        mutation_probability = trial.suggest_float(
            "mutation_probability", 0.0, 1.0, step=0.001
        )
        mutation_size = trial.suggest_float("mutation_size", 0.001, 2.0, step=0.001)
        select_percent = trial.suggest_float("select_percent", 0.1, 0.9, step=0.01)
        # tolerance = trial.suggest_int("tolerance_exp", -5, -2)
        # tolerance = 10**tolerance
        tolerance = 1e-5

        args += [
            "CNE",
            f"--population_size={population_size}",
            f"--max_generations={max_generations}",
            f"--mutation_probability={mutation_probability}",
            f"--mutation_size={mutation_size}",
            f"--select_percent={select_percent}",
            f"--tolerance={tolerance}",
        ]
    elif optim_type == "Differential Evolution":
        population_size = trial.suggest_int("population_size", 100, 10000, step=10)
        max_generations = trial.suggest_int("max_generations", 10, 10000, step=10)
        crossover_rate = trial.suggest_float("crossover_rate", 0.0, 1.0, step=0.001)
        differential_weight = trial.suggest_float(
            "differential_weight", 0.0, 2.0, step=0.001
        )
        # tolerance = trial.suggest_int("tolerance_exp", -5, -2)
        # tolerance = 10**tolerance
        tolerance = 1e-5

        args += [
            "DifferentialEvolution",
            f"--population_size={population_size}",
            f"--max_generations={max_generations}",
            f"--crossover_rate={crossover_rate}",
            f"--differential_weight={differential_weight}",
            f"--tolerance={tolerance}",
        ]
    elif optim_type == "PSO":
        num_particles = trial.suggest_int("num_particles", 5, 500)
        max_iterations = trial.suggest_int("max_iterations", 20, 5000, step=10)
        horizon_size = trial.suggest_int("horizon_size", 10, 500, step=10)
        max_iterations += horizon_size  # PSO has a constraint that max_iterations should be larger than horizon_size
        exploitation_factor = trial.suggest_float(
            "exploitation_factor", 2.0, 3.0, step=0.001
        )
        exploration_factor = trial.suggest_float(
            "exploration_factor", 2.0, 3.0, step=0.001
        )
        tolerance = trial.suggest_int("tolerance_exp", -5, -2)
        tolerance = 10**tolerance

        args += [
            "PSO",
            f"--num_particles={num_particles}",
            f"--max_iterations={max_iterations}",
            f"--horizon_size={horizon_size}",
            f"--exploitation_factor={exploitation_factor}",
            f"--exploration_factor={exploration_factor}",
            f"--tolerance={tolerance}",
        ]
    elif optim_type == "CMA-ES":
        population_size = trial.suggest_int("population_size", 0, 1000, step=10)
        max_iterations = trial.suggest_int("max_iterations", 1, 10)
        max_iterations = 10**max_iterations
        tolerance = trial.suggest_int("tolerance_exp", -5, -2)
        tolerance = 10**tolerance
        step_size = trial.suggest_float("step_size", 0, 2.0, step=0.001)

        args += [
            "CMAES",
            f"--population_size={population_size}",
            f"--max_iterations={max_iterations}",
            f"--tolerance={tolerance}",
            f"--step_size={step_size}",
        ]

    result = run_optimization(args, optim_type, timeout=timeout)

    loss = [result["best_loss_colorless"]]
    time = [result["elapsed_time"]]

    # Run the tool multiple time and averag the results
    REPEAT = 5
    for _ in range(REPEAT - 1):
        result_tmp = run_optimization(args, optim_type, timeout=timeout)
        result["best_loss_colorless"] += result_tmp["best_loss_colorless"]
        result["best_loss_spectrum"] += result_tmp["best_loss_spectrum"]
        result["elapsed_time"] += result_tmp["elapsed_time"]
        result["total_evaluations_colorless"] += result_tmp[
            "total_evaluations_colorless"
        ]
        result["total_evaluations_spectrum"] += result_tmp["total_evaluations_spectrum"]
        loss.append(result_tmp["best_loss_colorless"])
        time.append(result_tmp["elapsed_time"])

    result["best_loss_colorless"] /= REPEAT
    result["best_loss_spectrum"] /= REPEAT
    result["elapsed_time"] /= REPEAT
    result["total_evaluations_colorless"] /= REPEAT
    result["total_evaluations_spectrum"] /= REPEAT

    trial.set_user_attr("elapsed_time", result["elapsed_time"])
    trial.set_user_attr(
        "total_evaluations_spectrum", result["total_evaluations_spectrum"]
    )
    trial.set_user_attr(
        "total_evaluations_colorless", result["total_evaluations_colorless"]
    )
    trial.set_user_attr("log_string", result["log_string"])
    trial.set_user_attr("cmd_args", args)
    trial.set_user_attr("losses_colorless", loss)
    trial.set_user_attr("times", time)

    return result["best_loss_colorless"], result["elapsed_time"]


# class OptimizationTask(object):
#     def __init__(self, optim_type):
#         self.optim_type = optim_type


#     def __call__(self, _):
def task(optim_type):
    # sampler = optuna.samplers.TPESampler()
    print(f"Starting optimization with {optim_type}...")
    sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()

    file_path = f"./optuna_journal_storage_{sampler.__class__.__name__}.log"

    tag = "Colorless" if COLORLESS_ONLY else "Spectrum"

    study = optuna.create_study(
        sampler=sampler,
        storage=JournalStorage(JournalFileBackend(file_path)),
        study_name=f"{optim_type} - {tag}/time ({sampler.__class__.__name__}) - (N={FDN_SIZE})",
        load_if_exists=True,
        directions=(["minimize", "minimize"]),
    )
    study.set_metric_names(["loss", "time"])

    if optim_type == "Adam":
        study.enqueue_trial(
            {
                "step_size": 0.1,
                "beta1": 0.9,
                "beta2": 0.999,
                "gradient_delta_exp": -3,
                "tolerance_exp": -4,
            }
        )
    elif optim_type == "SPSA":
        study.enqueue_trial(
            {
                "alpha": 0.602,
                "gamma": 0.101,
                "step_size": 0.16,
                "eval_step_size": 0.3,
                "max_iterations": 6,
                "tolerance_exp": -5,
            }
        )
    elif optim_type == "Simulated Annealing":
        study.enqueue_trial(
            {
                "max_iterations": 7,
                "initial_temperature": 10000,
                "init_moves": 1000,
                "move_ctrl_sweep": 100,
                "tolerance_exp": -5,
                "max_tolerance_sweep": 3,
                "max_move_coeff": 20,
                "init_move_coeff": 0.3,
                "gain": 0.3,
            }
        )
    elif optim_type == "CNE":
        study.enqueue_trial(
            {
                "population_size": 500,
                "max_generations": 5000,
                "mutation_probability": 0.1,
                "mutation_size": 0.02,
                "select_percent": 0.2,
                "tolerance_exp": -5,
            }
        )
    elif optim_type == "Differential Evolution":
        study.enqueue_trial(
            {
                "population_size": 100,
                "max_generations": 2000,
                "crossover_rate": 0.6,
                "differential_weight": 0.8,
                "tolerance_exp": -5,
            }
        )
    elif optim_type == "PSO":
        study.enqueue_trial(
            {
                "num_particles": 64,
                "max_iterations": 3000,
                "horizon_size": 350,
                "exploitation_factor": 2.05,
                "exploration_factor": 2.05,
            }
        )
    elif optim_type == "L-BFGS":
        study.enqueue_trial(
            {
                "num_basis": 10,
                "max_iterations": 1e5,
                "wolfe": 0.9,
                "min_gradient_norm_exp": -6,
                "factor_exp": -15,
                "max_line_search": 50,
                "min_step": -20,
                "max_step": 20,
                "gradient_delta_exp": -3,
            }
        )
    elif optim_type == "Gradient Descent":
        study.enqueue_trial(
            {
                "step_size": 1.0,
                "max_iterations": 6,
                "tolerance_exp": -5,
                "kappa": 0.2,
                "phi": 0.8,
                "momentum": 0.5,
                "min_gain": -8,
                "gradient_delta_exp": -3,
            }
        )
    elif optim_type == "CMA-ES":
        study.enqueue_trial(
            {
                "population_size": 0,
                "max_iterations": 4,
                "tolerance_exp": -5,
                "step_size": 0,
            }
        )

    study.optimize(
        lambda trial: objective(optim_type, trial),
        n_trials=500,
        gc_after_trial=True,
    )


if __name__ == "__main__":

    # Check if the executable exists
    if not path_to_exe.is_file():
        print(f"Executable not found at {path_to_exe}")
        exit(1)

    terminate = False

    # optimizers = OPTIM_TYPES[1:]
    optimizers = ["Gradient Descent"]

    N_PROCESSES = len(optimizers)

    with Pool(processes=N_PROCESSES) as pool:
        pool.map(task, optimizers)
