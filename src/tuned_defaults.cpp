#include "tuned_defaults.h"

#include <cassert>

// Tuned hyperparameters transcribed from the two shipped campaign configurations:
//   app/config-colorless.toml
//   app/config-rir-matching.toml
//
// The provenance comment from each configuration block is preserved so the accept/reject rationale
// behind a value is not lost when it is read here rather than in the TOML. Keys a campaign does not
// set are deliberately omitted, leaving the struct initializer in optimizer.h as the single
// definition of an untuned value.

namespace fdn_optimization
{
namespace
{

AdamParameters ColorlessAdam()
{
    // Research-defined tuned candidate rejected by the held-out 5% gate.
    // Candidate 0.037996 vs incumbent 0.037410 (-1.6% improvement).
    // Retaining shipped incumbent.
    AdamParameters parameters;
    parameters.step_size = 0.442;
    parameters.beta1 = 0.9;
    parameters.beta2 = 0.851;
    parameters.tolerance = 1e-5;
    parameters.gradient_delta = 1e-2;
    return parameters;
}

AdamParameters MatchingAdam()
{
    // Tuned on concert_hall; held-out room medians:
    // concert_hall=0.020940, meeting_room=0.007909, office_lobby=0.018648.
    // Improvement over concert-hall incumbent: 83.8%.
    AdamParameters parameters;
    parameters.step_size = 0.01415682760783907;
    parameters.beta1 = 0.8817131289468394;
    parameters.beta2 = 0.9290539231414752;
    parameters.tolerance = 1.8551684526113878e-07;
    parameters.max_iterations = 2327;
    parameters.gradient_delta = 0.01;
    return parameters;
}

SPSAParameters ColorlessSPSA()
{
    // Tuned: research-defined search space, colorless N=8, thesis init, 4 threads.
    // Held-out median 0.043462 vs incumbent 0.065802 (34.0% better): ACCEPTED.
    SPSAParameters parameters;
    parameters.step_size = 1.1727627377482301;
    parameters.evaluationStepSize = 0.05949918350903441;
    parameters.alpha = 0.22755572069225438;
    parameters.gamma = 0.07784861135731835;
    parameters.max_iterations = 200578;
    parameters.tolerance = 1e-8;
    return parameters;
}

SPSAParameters MatchingSPSA()
{
    // Tuned candidate rejected: no valid incumbent (default failed).
    // Retaining shipped incumbent.
    SPSAParameters parameters;
    parameters.alpha = 0.2880938193488607;
    parameters.gamma = 0.7373144814549601;
    parameters.step_size = 1.6640000000000001;
    parameters.evaluationStepSize = 1.132721870064672;
    parameters.max_iterations = 1000000;
    parameters.tolerance = 1e-5;
    return parameters;
}

BlockSPSAParameters ColorlessBlockSPSA()
{
    // Tuned: research-defined search space, colorless N=8, thesis init, 4 threads.
    // Held-out median 0.041372 vs incumbent 0.048993 (15.6% better): ACCEPTED.
    BlockSPSAParameters parameters;
    parameters.step_size = 51.36856103655465;
    parameters.evaluation_step_size = 0.1274603915360556;
    parameters.stability_constant = 15.55350824284848;
    parameters.alpha = 0.44541503073811406;
    parameters.gamma = 0.26535782766381644;
    parameters.directions_per_block = 4;
    parameters.mode = BlockSPSAMode::SnapshotSweepAll;
    parameters.block_strategy = ParameterBlockStrategy::Semantic;
    parameters.random_schedule = RandomBlockSchedule::ShuffledSweep;
    parameters.probe_radius_normalization = ProbeRadiusNormalization::None;
    parameters.accepted_evaluation_interval = 1;
    parameters.max_step_norm = 1.0;
    parameters.tolerance = 1e-5;
    return parameters;
}

BlockSPSAParameters MatchingBlockSPSA()
{
    // Tuned on concert_hall; held-out room medians:
    // concert_hall=0.016379, meeting_room=0.009704, office_lobby=0.021158.
    // Improvement over concert-hall incumbent: 46.9%.
    BlockSPSAParameters parameters;
    parameters.step_size = 8.72124107507604;
    parameters.evaluation_step_size = 0.03652760582085782;
    parameters.stability_constant = 161.88023205885085;
    parameters.alpha = 0.4074201574707456;
    parameters.gamma = 0.4755784907490115;
    parameters.max_step_norm = 1.0349578577007343;
    parameters.directions_per_block = 1;
    parameters.tolerance = 6.709425909797098e-05;
    parameters.stall_window = 320;
    parameters.max_iterations = 7578;
    parameters.mode = BlockSPSAMode::SnapshotSweepAll;
    parameters.block_strategy = ParameterBlockStrategy::Semantic;
    parameters.random_schedule = RandomBlockSchedule::ShuffledSweep;
    parameters.probe_radius_normalization = ProbeRadiusNormalization::None;
    parameters.accepted_evaluation_interval = 1;
    parameters.block_scales = {
        {.scale_class = BlockScaleClass::Attenuation, .a_scale = 2.5140841111296766, .c_scale = 0.9711107286375275},
        {.scale_class = BlockScaleClass::Tone, .a_scale = 0.40045493705374097, .c_scale = 0.7862662730066963},
        {.scale_class = BlockScaleClass::OverallGain, .a_scale = 0.3203531808496319, .c_scale = 0.14115904814050217},
    };
    return parameters;
}

SimulatedAnnealingParameters ColorlessSimulatedAnnealing()
{
    // Tuned: research-defined search space, colorless N=8, thesis init, 4 threads.
    // Held-out median 0.046764 vs incumbent 0.054726 (14.5% better): ACCEPTED.
    SimulatedAnnealingParameters parameters;
    parameters.initial_temperature = 0.36035107704955066;
    parameters.init_moves = 454;
    parameters.move_ctrl_sweep = 17;
    parameters.max_tolerance_sweep = 6;
    parameters.max_move_coef = 46.245108465477124;
    parameters.init_move_coef = 2.3170323742191057;
    parameters.gain = 0.7188287156432291;
    parameters.max_iterations = 1000000;
    parameters.tolerance = 1e-15;
    return parameters;
}

SimulatedAnnealingParameters MatchingSimulatedAnnealing()
{
    // Tuned on concert_hall; held-out room medians:
    // concert_hall=0.055041, meeting_room=0.014459, office_lobby=0.031696.
    // Improvement over concert-hall incumbent: 52.2%.
    SimulatedAnnealingParameters parameters;
    parameters.initial_temperature = 0.5113007199819358;
    parameters.init_moves = 206;
    parameters.move_ctrl_sweep = 15;
    parameters.max_tolerance_sweep = 4;
    parameters.max_move_coef = 24.58623391467423;
    parameters.init_move_coef = 0.025632542951562563;
    parameters.gain = 0.7188287156432291;
    parameters.tolerance = 0.00014790552196423276;
    parameters.max_iterations = 124961;
    return parameters;
}

CNEParameters ColorlessCNE()
{
    // Research-defined tuned candidate rejected by the held-out 5% gate.
    // Candidate 0.069530 vs incumbent 0.070673 (1.6% improvement).
    // Retaining shipped incumbent.
    CNEParameters parameters;
    parameters.population_size = 5200;
    parameters.max_generations = 7470;
    parameters.mutation_probability = 0.737;
    parameters.mutation_size = 0.171;
    parameters.select_percent = 0.72;
    parameters.tolerance = 1e-5;
    return parameters;
}

CNEParameters MatchingCNE()
{
    // Tuned on concert_hall; held-out room medians:
    // concert_hall=0.076795, meeting_room=0.040283, office_lobby=0.140753.
    // Improvement over concert-hall incumbent: 24.3%.
    CNEParameters parameters;
    parameters.population_size = 82;
    parameters.max_generations = 4165;
    parameters.mutation_probability = 0.011838288500026257;
    parameters.mutation_size = 0.12273397513680598;
    parameters.select_percent = 0.29680584290545414;
    parameters.tolerance = 0.0001994968503978872;
    return parameters;
}

DifferentialEvolutionParameters ColorlessDifferentialEvolution()
{
    // Tuned: research-defined search space, colorless N=8, thesis init, 4 threads.
    // Held-out median 0.058079 vs incumbent 0.066494 (12.7% better): ACCEPTED.
    DifferentialEvolutionParameters parameters;
    parameters.population_size = 491;
    parameters.max_generation = 4256;
    parameters.crossover_rate = 0.7847035452441409;
    parameters.differential_weight = 0.5012929168816291;
    parameters.tolerance = 0.0003786985872223152;
    return parameters;
}

DifferentialEvolutionParameters MatchingDifferentialEvolution()
{
    // Tuned on concert_hall; held-out room medians:
    // concert_hall=0.031879, meeting_room=0.022089, office_lobby=0.056616.
    // Improvement over concert-hall incumbent: 52.1%.
    DifferentialEvolutionParameters parameters;
    parameters.population_size = 634;
    parameters.max_generation = 442;
    parameters.crossover_rate = 0.9767983284150052;
    parameters.differential_weight = 0.7200634774391111;
    parameters.tolerance = 0.00042736022405851914;
    return parameters;
}

PSOParameters ColorlessPSO()
{
    // Tuned: research-defined search space, colorless N=8, thesis init, 4 threads.
    // Held-out median 0.042393 vs incumbent 0.049140 (13.7% better): ACCEPTED.
    PSOParameters parameters;
    parameters.num_particles = 88;
    parameters.max_iterations = 4951;
    parameters.tolerance = 7.355700350218312e-05;
    parameters.horizon_size = 2130;
    parameters.exploitation_factor = 1.9157717549374635;
    parameters.exploration_factor = 2.2087542742713775;
    return parameters;
}

PSOParameters MatchingPSO()
{
    // Tuned on concert_hall; held-out room medians:
    // concert_hall=0.025024, meeting_room=0.021368, office_lobby=0.050135.
    // Improvement over concert-hall incumbent: 8.3%.
    PSOParameters parameters;
    parameters.num_particles = 45;
    parameters.max_iterations = 1991;
    parameters.tolerance = 1.413335407803018e-06;
    parameters.horizon_size = 221;
    parameters.exploitation_factor = 2.0818751258045385;
    parameters.exploration_factor = 2.1102229100722045;
    return parameters;
}

L_BFGSParameters ColorlessLBFGS()
{
    // Tuned: research-defined search space, colorless N=8, thesis init, 4 threads.
    // Held-out median 0.046892 vs incumbent 0.051806 (9.5% better): ACCEPTED.
    L_BFGSParameters parameters;
    parameters.num_basis = 28;
    parameters.max_iterations = 155;
    parameters.wolfe = 0.981183364797702;
    parameters.max_line_search_trials = 50;
    parameters.gradient_delta = 0.00032242935935624875;
    parameters.min_gradient_norm = 1e-6;
    parameters.factor = 1e-15;
    parameters.min_step = 1e-20;
    parameters.max_step = 1e20;
    return parameters;
}

L_BFGSParameters MatchingLBFGS()
{
    // Tuned on concert_hall; held-out room medians:
    // concert_hall=0.010905, meeting_room=0.007971, office_lobby=0.018435.
    // Improvement over concert-hall incumbent: 59.7%.
    L_BFGSParameters parameters;
    parameters.num_basis = 13;
    parameters.max_iterations = 688;
    parameters.wolfe = 0.8546366488678708;
    parameters.max_line_search_trials = 53;
    parameters.min_gradient_norm = 1.678861289003167e-08;
    parameters.factor = 1.7410976911860362e-13;
    parameters.gradient_delta = 0.005087368202829873;
    parameters.min_step = 1e-20;
    parameters.max_step = 1e20;
    return parameters;
}

GradientDescentParameters ColorlessGradientDescent()
{
    // Tuned: research-defined search space, colorless N=8, thesis init, 4 threads.
    // Held-out median 0.042894 vs incumbent 0.053873 (20.4% better): ACCEPTED.
    GradientDescentParameters parameters;
    parameters.step_size = 5.23434635797284;
    parameters.kappa = 0.7132578918476734;
    parameters.phi = 0.08375222229660942;
    parameters.momentum = 0.3078781982633325;
    parameters.min_gain = 2.316819665320155e-06;
    parameters.gradient_delta = 0.007608971111929962;
    parameters.tolerance = 1e-5;
    parameters.max_iterations = 100000000;
    return parameters;
}

GradientDescentParameters MatchingGradientDescent()
{
    // Trust-region Momentum Delta-Bar-Delta, tuned after saturation investigation.
    // concert_hall=0.013301, meeting_room=0.008393, office_lobby=0.018968.
    // Improvement over concert-hall incumbent: 89.8%.
    GradientDescentParameters parameters;
    parameters.step_size = 20;
    parameters.kappa = 0.004837998433178365;
    parameters.phi = 0.6904292600926617;
    parameters.momentum = 0.4134871604209004;
    parameters.min_gain = 4.5037914968952005e-7;
    parameters.gradient_delta = 0.01;
    parameters.max_step_norm = 1.0;
    parameters.tolerance = 1e-8;
    parameters.max_iterations = 5000;
    return parameters;
}

CMAESParameters ColorlessCMAES()
{
    // Research-defined tuned candidate rejected by the held-out 5% gate.
    // Candidate 0.042378 vs incumbent 0.044342 (4.4% improvement).
    // Retaining shipped incumbent.
    CMAESParameters parameters;
    parameters.population_size = 10;
    parameters.max_iterations = 1000000000;
    parameters.tolerance = 1e-4;
    parameters.step_size = 0.108;
    return parameters;
}

CMAESParameters MatchingCMAES()
{
    // Tuned after fixing parameter-aware CMA-ES matching bounds.
    // concert_hall=0.017477, meeting_room=0.011753, office_lobby=0.034304.
    // Improvement over pre-fix concert-hall incumbent: 99.9%.
    CMAESParameters parameters;
    parameters.population_size = 10;
    parameters.step_size = 0.108;
    parameters.tolerance = 0.0001;
    parameters.max_iterations = 200000;
    return parameters;
}

RandomSearchParameters TunedRandomSearch()
{
    // Untuned trivial baseline; time_limit is the comparison horizon, not a tuning parameter.
    // Both campaigns ship the same value.
    RandomSearchParameters parameters;
    parameters.time_limit_seconds = 10.0;
    return parameters;
}

} // namespace

OptimizationAlgoParams TunedDefaultsFor(OptimizationAlgoType algorithm, OptimizationObjective objective)
{
    const bool matching = objective == OptimizationObjective::RirMatching;
    switch (algorithm)
    {
    case OptimizationAlgoType::SPSA:
        return matching ? MatchingSPSA() : ColorlessSPSA();
    case OptimizationAlgoType::BlockSPSA:
        return matching ? MatchingBlockSPSA() : ColorlessBlockSPSA();
    case OptimizationAlgoType::SimulatedAnnealing:
        return matching ? MatchingSimulatedAnnealing() : ColorlessSimulatedAnnealing();
    case OptimizationAlgoType::DifferentialEvolution:
        return matching ? MatchingDifferentialEvolution() : ColorlessDifferentialEvolution();
    case OptimizationAlgoType::PSO:
        return matching ? MatchingPSO() : ColorlessPSO();
    case OptimizationAlgoType::RandomSearch:
        return TunedRandomSearch();
    case OptimizationAlgoType::CMAES:
        return matching ? MatchingCMAES() : ColorlessCMAES();
    case OptimizationAlgoType::CNE:
        return matching ? MatchingCNE() : ColorlessCNE();
    case OptimizationAlgoType::Adam:
        return matching ? MatchingAdam() : ColorlessAdam();
    case OptimizationAlgoType::L_BFGS:
        return matching ? MatchingLBFGS() : ColorlessLBFGS();
    case OptimizationAlgoType::GradientDescent:
        return matching ? MatchingGradientDescent() : ColorlessGradientDescent();
    case OptimizationAlgoType::Count:
    default:
        break;
    }

    // Count is not an algorithm and every real alternative is handled above. Returning a variant
    // whose alternative disagrees with the request is the failure mode this API exists to prevent,
    // so fail loudly in debug builds rather than silently substituting an unrelated optimizer.
    assert(false && "TunedDefaultsFor called with an unhandled OptimizationAlgoType");
    return TunedRandomSearch();
}

TunedLossDefaults TunedLossDefaultsFor(OptimizationObjective objective)
{
    // Both campaigns currently ship identical weights; the parameter keeps callers correct if the
    // two ever diverge.
    static_cast<void>(objective);
    return TunedLossDefaults{
        .spectral_flatness_weight = 1.0,
        .sparsity_weight = 0.5,
        .power_envelope_weight = 0.0,
        .edc_weight = 0.1,
        .mel_edr_weight = 1.0,
        .weighted_edr_weight = 0.0,
    };
}

TunedMatchingSettings TunedMatchingDefaults()
{
    // matching_filter_type = "3band", matching_initialization = "random".
    return TunedMatchingSettings{
        .attenuation_filter_type = OptimizationParamType::AttenuationFilters_3Band,
        .initialization = MatchingInitialization::SeededRandom,
    };
}

} // namespace fdn_optimization
