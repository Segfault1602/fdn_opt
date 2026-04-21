#pragma once

#include <sffdn/sffdn.h>

#include <quill/LogMacros.h>
#include <quill/Logger.h>

#include "optim_types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <variant>

namespace fdn_optimization
{

enum class OptimizationStatus : uint8_t
{
    Ready,
    StartRequested,
    Running,
    CancelRequested,
    Completed,
    Canceled,
    Failed
};

struct AdamParameters
{
    float step_size = 0.442;
    float beta1 = 0.9;
    float beta2 = 0.851;
    float learning_rate_decay = 1.0;
    int decay_step_size = 1;
    int epoch_restarts = 100;
    int max_restarts = 0;
    float tolerance = 1e-5;

    double gradient_delta = 1e-2;
};

struct L_BFGSParameters
{
    size_t num_basis = 292;
    size_t max_iterations = 1301945;
    double wolfe = 0.949;
    double min_gradient_norm = 1e-6;
    double factor = 1e-15;
    size_t max_line_search_trials = 31;
    double min_step = 1e-20;
    double max_step = 1e20;

    double gradient_delta = 1e-4;
};

struct GradientDescentParameters
{
    double step_size = 2;
    size_t max_iterations = 10000000000;
    double tolerance = 1e-5;

    double kappa = 0.01;
    double phi = 0.01;
    double momentum = 0.812;
    double min_gain = 1e-2;

    double gradient_delta = 1e-1;
};

struct SPSAParameters
{
    double alpha = 0.2880938193488607;
    double gamma = 0.7373144814549601;
    double step_size = 1.6640000000000001;
    double evaluationStepSize = 1.132721870064672;
    size_t max_iterations = 1000000;
    double tolerance = 1e-5;
};

struct SimulatedAnnealingParameters
{
    size_t max_iterations = 1000000;
    double initial_temperature = 5;
    size_t init_moves = 10;
    size_t move_ctrl_sweep = 1;
    size_t max_tolerance_sweep = 30;
    double max_move_coef = 30;
    double init_move_coef = 2;
    double gain = 1.8;
    double tolerance = 1e-5;
};

struct CNEParameters
{
    size_t population_size = 5200;
    size_t max_generations = 7470;
    double mutation_probability = 0.737;
    double mutation_size = 0.171;
    double select_percent = 0.72;
    double tolerance = 1e-5;
};

struct DifferentialEvolutionParameters
{
    size_t population_size = 3390;
    size_t max_generation = 8960;
    double crossover_rate = 1.0;
    double differential_weight = 0.81;
    double tolerance = 1e-5;
};

struct PSOParameters
{
    size_t num_particles = 49;
    size_t max_iterations = 4060;
    size_t horizon_size = 410;
    double exploitation_factor = 2.123;
    double exploration_factor = 2.05;
    double tolerance = 1e-5;
};

struct RandomSearchParameters
{
    double time_limit_seconds = 10.0;
};

struct CMAESParameters
{
    size_t population_size = 10;
    size_t max_iterations = 1000000000;
    double tolerance = 1e-5;
    double step_size = 0 / 108;
};

enum class OptimizationAlgoType : uint8_t
{
    SPSA,
    SimulatedAnnealing,
    DifferentialEvolution,
    PSO,
    RandomSearch,
    CMAES,
    CNE,
    // Below here use gradient information
    Adam,
    L_BFGS,
    GradientDescent,
    Count,
};

constexpr const char* OptimizationAlgoTypeToString(OptimizationAlgoType type)
{
    switch (type)
    {
    case OptimizationAlgoType::Adam:
        return "Adam";
    case OptimizationAlgoType::SPSA:
        return "SPSA";
    case OptimizationAlgoType::SimulatedAnnealing:
        return "Simulated Annealing";
    case OptimizationAlgoType::CNE:
        return "CNE";
    case OptimizationAlgoType::DifferentialEvolution:
        return "Differential Evolution";
    case OptimizationAlgoType::PSO:
        return "Particle Swarm Optimization";
    case OptimizationAlgoType::RandomSearch:
        return "Random Search";
    case OptimizationAlgoType::L_BFGS:
        return "L-BFGS";
    case OptimizationAlgoType::GradientDescent:
        return "Gradient Descent";
    case OptimizationAlgoType::CMAES:
        return "CMA-ES";
    default:
        return "Unknown";
    }
}

using OptimizationAlgoParams =
    std::variant<AdamParameters, SPSAParameters, SimulatedAnnealingParameters, DifferentialEvolutionParameters,
                 PSOParameters, RandomSearchParameters, L_BFGSParameters, GradientDescentParameters, CMAESParameters,
                 CNEParameters>;

struct OptimizationInfo
{
    std::vector<OptimizationParamType> parameters_to_optimize;
    sfFDN::FDNConfig initial_fdn_config;
    uint32_t ir_size;
    fdn_optimization::GradientMethod gradient_method = fdn_optimization::GradientMethod::CentralDifferences;

    // Colorless loss weights
    double spectral_flatness_weight = 1.0;
    double sparsity_weight = 1.0;
    double power_envelope_weight = 0.0;

    // RIR match loss weights
    double edc_weight = 1.0;
    double mel_edr_weight = 1.0;
    double weighted_edr_weight = 0.0;

    uint32_t mel_edr_fft_length = 4096;
    uint32_t mel_edr_hop_size = 128;
    uint32_t mel_edr_window_size = 1024;
    uint32_t mel_edr_num_bands = 64;

    std::vector<float> target_rir;
    std::vector<float> early_fir;

    OptimizationAlgoParams optimizer_params;
};

struct OptimizationProgressInfo
{
    std::chrono::duration<double> elapsed_time;
    uint32_t evaluation_count;
    std::vector<std::vector<double>> loss_history;
};

struct OptimizationResult
{
    sfFDN::FDNConfig initial_fdn_config;
    sfFDN::FDNConfig optimized_fdn_config;
    std::chrono::duration<double> total_time;
    uint32_t total_evaluations;
    std::vector<std::vector<double>> loss_history;
    std::vector<std::string> loss_names;
    double best_loss;
};

class OptimCallback;

class FDNOptimizer
{
  public:
    FDNOptimizer(quill::Logger* logger, bool verbose = false);
    ~FDNOptimizer();

    void StartOptimization(OptimizationInfo& info);
    void CancelOptimization();
    void ResetStatus();

    OptimizationStatus GetStatus() const;
    OptimizationProgressInfo GetProgress();

    OptimizationResult GetResult();

  private:
    void ThreadProc(std::stop_token stop_token, OptimizationInfo info);

    quill::Logger* logger_;
    bool verbose_;
    std::atomic<OptimizationStatus> status_;

    std::chrono::steady_clock::time_point start_time_;
    std::jthread thread_;
    std::mutex mutex_;

    sfFDN::FDNConfig optimized_config_;
    OptimizationResult optimization_result_;

    std::unique_ptr<OptimCallback> optim_callback_;
};
} // namespace fdn_optimization