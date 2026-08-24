#include "optimizer_cli.h"

#include <map>
#include <utility>

namespace
{

template <typename Params>
void SelectOptimizer(fdn_opt_app::OptimizerCliOptions& options, std::string name, const Params& params)
{
    options.selected_name = std::move(name);
    options.selected_params = params;
}

} // namespace

namespace fdn_opt_app
{

void RegisterOptimizerSubcommands(CLI::App& app, OptimizerCliOptions& options)
{
    auto* adam = app.add_subcommand("Adam", "Use Adam optimization algorithm");
    adam->add_option("--step_size", options.adam.step_size, "Step size for Adam optimizer");
    adam->add_option("--beta1", options.adam.beta1, "Beta1 parameter for Adam optimizer");
    adam->add_option("--beta2", options.adam.beta2, "Beta2 parameter for Adam optimizer");
    adam->add_option("--tolerance", options.adam.tolerance, "Tolerance for Adam optimizer");
    adam->add_option("--gradient_delta", options.adam.gradient_delta, "Gradient delta for Adam optimizer");
    adam->add_option("--max_iterations", options.adam.max_iterations, "Maximum Adam iterations");
    adam->callback([&options]() { SelectOptimizer(options, "Adam", options.adam); });

    auto* spsa = app.add_subcommand("SPSA", "Use SPSA optimization algorithm");
    spsa->add_option("--alpha", options.spsa.alpha, "Alpha parameter for SPSA optimizer");
    spsa->add_option("--gamma", options.spsa.gamma, "Gamma parameter for SPSA optimizer");
    spsa->add_option("--step_size", options.spsa.step_size, "Step size for SPSA optimizer");
    spsa->add_option("--evaluation_step_size", options.spsa.evaluationStepSize,
                     "Evaluation step size for SPSA optimizer");
    spsa->add_option("--max_iterations", options.spsa.max_iterations, "Maximum iterations for SPSA optimizer");
    spsa->add_option("--tolerance", options.spsa.tolerance, "Tolerance for SPSA optimizer");
    spsa->callback([&options]() { SelectOptimizer(options, "SPSA", options.spsa); });

    auto* block_spsa = app.add_subcommand("BlockSPSA", "Use block-coordinate SPSA optimization");
    block_spsa->add_option("--mode", options.block_spsa.mode, "BlockSPSA update mode")
        ->transform(CLI::CheckedTransformer(std::map<std::string, fdn_optimization::BlockSPSAMode>{
            {"snapshot", fdn_optimization::BlockSPSAMode::SnapshotSweepAll},
            {"random-one", fdn_optimization::BlockSPSAMode::RandomOne},
        }));
    block_spsa->add_option("--block_strategy", options.block_spsa.block_strategy, "Parameter block strategy")
        ->transform(CLI::CheckedTransformer(std::map<std::string, fdn_optimization::ParameterBlockStrategy>{
            {"semantic", fdn_optimization::ParameterBlockStrategy::Semantic},
            {"fixed", fdn_optimization::ParameterBlockStrategy::FixedContiguous},
        }));
    block_spsa->add_option("--random_schedule", options.block_spsa.random_schedule, "Random block schedule")
        ->transform(CLI::CheckedTransformer(std::map<std::string, fdn_optimization::RandomBlockSchedule>{
            {"shuffled", fdn_optimization::RandomBlockSchedule::ShuffledSweep},
            {"uniform", fdn_optimization::RandomBlockSchedule::IndependentUniform},
        }));
    block_spsa
        ->add_option("--three_band_grouping", options.block_spsa.three_band_grouping,
                     "Semantic grouping for independent three-band attenuation")
        ->transform(CLI::CheckedTransformer(std::map<std::string, fdn_optimization::ThreeBandBlockGrouping>{
            {"channel-triplets", fdn_optimization::ThreeBandBlockGrouping::ChannelTriplets},
            {"frequency-bands", fdn_optimization::ThreeBandBlockGrouping::FrequencyBands},
        }));
    block_spsa->add_option("--block_size", options.block_spsa.contiguous_block_size, "Fixed contiguous block size");
    block_spsa
        ->add_option("--probe_radius_normalization", options.block_spsa.probe_radius_normalization,
                     "Scales the probe amplitude by block dimension")
        ->transform(CLI::CheckedTransformer(std::map<std::string, fdn_optimization::ProbeRadiusNormalization>{
            {"none", fdn_optimization::ProbeRadiusNormalization::None},
            {"sqrt-dim", fdn_optimization::ProbeRadiusNormalization::SqrtDimension},
        }));
    block_spsa->add_option("--block_scale", options.block_spsa_scale_specs,
                           "Per-class gain scales as <class>:<a_scale>:<c_scale>; repeatable. Classes: default, "
                           "gains_in, gains_out, matrix, attenuation, tone, overall_gain");
    block_spsa->add_option("--accepted_eval_interval", options.block_spsa.accepted_evaluation_interval,
                           "Evaluate the accepted point every N updates instead of every update");
    block_spsa->add_option("--max_step_norm", options.block_spsa.max_step_norm,
                           "Cap on the Euclidean norm of one coordinate update; 0 disables it");
    block_spsa->add_option("--stall_window", options.block_spsa.stall_window,
                           "Evaluated accepted points over which the best loss must improve by more than "
                           "--tolerance; 0 disables early stopping");
    block_spsa->add_option("--directions_per_block", options.block_spsa.directions_per_block,
                           "Perturbation directions averaged per block");
    block_spsa->add_option("--alpha", options.block_spsa.alpha, "Step-size decay exponent");
    block_spsa->add_option("--gamma", options.block_spsa.gamma, "Perturbation decay exponent");
    block_spsa->add_option("--step_size", options.block_spsa.step_size, "Step-size scale");
    block_spsa->add_option("--evaluation_step_size", options.block_spsa.evaluation_step_size, "Perturbation scale");
    block_spsa->add_option("--stability_constant", options.block_spsa.stability_constant,
                           "Optional absolute gain stability constant");
    block_spsa->add_option("--max_iterations", options.block_spsa.max_iterations, "Maximum accepted BlockSPSA updates");
    block_spsa->add_option("--tolerance", options.block_spsa.tolerance,
                           "Relative best-loss improvement required across --stall_window evaluated points");
    block_spsa->callback([&options]() { SelectOptimizer(options, "BlockSPSA", options.block_spsa); });

    auto* simulated_annealing =
        app.add_subcommand("SimulatedAnnealing", "Use Simulated Annealing optimization algorithm");
    simulated_annealing->add_option("--max_iterations", options.simulated_annealing.max_iterations,
                                    "Maximum iterations for Simulated Annealing");
    simulated_annealing->add_option("--initial_temperature", options.simulated_annealing.initial_temperature,
                                    "Initial temperature for Simulated Annealing");
    simulated_annealing->add_option("--init_moves", options.simulated_annealing.init_moves,
                                    "Initial moves for Simulated Annealing");
    simulated_annealing->add_option("--move_ctrl_sweep", options.simulated_annealing.move_ctrl_sweep,
                                    "Move control sweep for Simulated Annealing");
    simulated_annealing->add_option("--max_tolerance_sweep", options.simulated_annealing.max_tolerance_sweep,
                                    "Max tolerance sweep for Simulated Annealing");
    simulated_annealing->add_option("--max_move_coeff", options.simulated_annealing.max_move_coef,
                                    "Max move coefficient for Simulated Annealing");
    simulated_annealing->add_option("--init_move_coeff", options.simulated_annealing.init_move_coef,
                                    "Initial move coefficient for Simulated Annealing");
    simulated_annealing->add_option("--gain", options.simulated_annealing.gain, "Gain for Simulated Annealing");
    simulated_annealing->add_option("--tolerance", options.simulated_annealing.tolerance,
                                    "Tolerance for Simulated Annealing");
    simulated_annealing->callback(
        [&options]() { SelectOptimizer(options, "SimulatedAnnealing", options.simulated_annealing); });

    auto* cne = app.add_subcommand("CNE", "Use CNE optimization algorithm");
    cne->add_option("--population_size", options.cne.population_size, "Population size for CNE optimizer");
    cne->add_option("--max_generations", options.cne.max_generations, "Maximum generations for CNE optimizer");
    cne->add_option("--mutation_probability", options.cne.mutation_probability,
                    "Mutation probability for CNE optimizer");
    cne->add_option("--mutation_size", options.cne.mutation_size, "Mutation size for CNE optimizer");
    cne->add_option("--select_percent", options.cne.select_percent, "Selection percentage for CNE optimizer");
    cne->add_option("--tolerance", options.cne.tolerance, "Tolerance for CNE optimizer");
    cne->callback([&options]() { SelectOptimizer(options, "CNE", options.cne); });

    auto* differential_evolution =
        app.add_subcommand("DifferentialEvolution", "Use Differential Evolution optimization algorithm");
    differential_evolution->add_option("--population_size", options.differential_evolution.population_size,
                                       "Population size for Differential Evolution optimizer");
    differential_evolution->add_option("--max_generations", options.differential_evolution.max_generation,
                                       "Maximum generations for Differential Evolution optimizer");
    differential_evolution->add_option("--crossover_rate", options.differential_evolution.crossover_rate,
                                       "Crossover rate for Differential Evolution optimizer");
    differential_evolution->add_option("--differential_weight", options.differential_evolution.differential_weight,
                                       "Differential weight for Differential Evolution optimizer");
    differential_evolution->add_option("--tolerance", options.differential_evolution.tolerance,
                                       "Tolerance for Differential Evolution optimizer");
    differential_evolution->callback(
        [&options]() { SelectOptimizer(options, "DifferentialEvolution", options.differential_evolution); });

    auto* pso = app.add_subcommand("PSO", "Use Particle Swarm Optimization algorithm");
    pso->add_option("--num_particles", options.pso.num_particles, "Number of particles for PSO optimizer");
    pso->add_option("--max_iterations", options.pso.max_iterations, "Maximum iterations for PSO optimizer");
    pso->add_option("--horizon_size", options.pso.horizon_size, "Horizon size for PSO optimizer");
    pso->add_option("--exploitation_factor", options.pso.exploitation_factor, "Exploitation factor for PSO optimizer");
    pso->add_option("--exploration_factor", options.pso.exploration_factor, "Exploration factor for PSO optimizer");
    pso->add_option("--tolerance", options.pso.tolerance, "Tolerance for PSO optimizer");
    pso->callback([&options]() { SelectOptimizer(options, "PSO", options.pso); });

    auto* lbfgs = app.add_subcommand("L-BFGS", "Use L-BFGS optimization algorithm");
    lbfgs->add_option("--num_basis", options.lbfgs.num_basis, "Number of basis vectors for L-BFGS optimizer");
    lbfgs->add_option("--max_iterations", options.lbfgs.max_iterations, "Maximum iterations for L-BFGS optimizer");
    lbfgs->add_option("--wolfe", options.lbfgs.wolfe, "Wolfe condition parameter for L-BFGS optimizer");
    lbfgs->add_option("--min_gradient_norm", options.lbfgs.min_gradient_norm,
                      "Minimum gradient norm for L-BFGS optimizer");
    lbfgs->add_option("--factor", options.lbfgs.factor, "Factor for L-BFGS optimizer");
    lbfgs->add_option("--max_line_search", options.lbfgs.max_line_search_trials,
                      "Maximum line search trials for L-BFGS optimizer");
    lbfgs->add_option("--min_step", options.lbfgs.min_step, "Minimum step size for L-BFGS optimizer");
    lbfgs->add_option("--max_step", options.lbfgs.max_step, "Maximum step size for L-BFGS optimizer");
    lbfgs->add_option("--gradient_delta", options.lbfgs.gradient_delta,
                      "Gradient delta for L-BFGS optimizer when optimizing filters");
    lbfgs->callback([&options]() { SelectOptimizer(options, "L-BFGS", options.lbfgs); });

    auto* gradient_descent = app.add_subcommand("GradientDescent", "Use Gradient Descent optimization algorithm");
    gradient_descent->add_option("--step_size", options.gradient_descent.step_size,
                                 "Step size for Gradient Descent optimizer");
    gradient_descent->add_option("--max_iterations", options.gradient_descent.max_iterations,
                                 "Maximum iterations for Gradient Descent optimizer");
    gradient_descent->add_option("--tolerance", options.gradient_descent.tolerance,
                                 "Tolerance for Gradient Descent optimizer");
    gradient_descent->add_option("--kappa", options.gradient_descent.kappa, "Kappa");
    gradient_descent->add_option("--phi", options.gradient_descent.phi, "Phi");
    gradient_descent->add_option("--momentum", options.gradient_descent.momentum, "Momemtum");
    gradient_descent->add_option("--min_gain", options.gradient_descent.min_gain,
                                 "Minimum gain for Gradient Descent optimizer");
    gradient_descent
        ->add_option("--gradient_delta", options.gradient_descent.gradient_delta,
                     "Gradient delta for Gradient Descent optimizer when optimizing filters")
        ->default_val(1e-2);
    gradient_descent->add_option("--max_step_norm", options.gradient_descent.max_step_norm,
                                 "Cap on one Momentum Delta-Bar-Delta update; 0 disables it");
    gradient_descent->callback([&options]() { SelectOptimizer(options, "GradientDescent", options.gradient_descent); });

    auto* cmaes = app.add_subcommand("CMAES", "Use CMA-ES optimization algorithm");
    cmaes->add_option("--population_size", options.cmaes.population_size, "Population size for CMA-ES optimizer");
    cmaes->add_option("--max_iterations", options.cmaes.max_iterations, "Maximum iterations for CMA-ES optimizer");
    cmaes->add_option("--tolerance", options.cmaes.tolerance, "Tolerance for CMA-ES optimizer");
    cmaes->add_option("--step_size", options.cmaes.step_size, "Step size for CMA-ES optimizer");
    cmaes->callback([&options]() { SelectOptimizer(options, "CMAES", options.cmaes); });

    auto* random_search = app.add_subcommand("RandomSearch", "Use Random Search optimization algorithm");
    random_search->add_option("--time_limit", options.random_search.time_limit_seconds,
                              "Time limit in seconds for Random Search optimizer");
    random_search->callback([&options]() { SelectOptimizer(options, "RandomSearch", options.random_search); });
}

} // namespace fdn_opt_app
