#pragma once

#include <sffdn/sffdn.h>

#include <quill/LogMacros.h>
#include <quill/Logger.h>

#include "audio_loss.h"
#include "optim_types.h"
#include "parameter_layout.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <variant>
#include <vector>

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
    size_t max_iterations = 1000000;
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
    double max_step_norm = 0.0;
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

// Selects how BlockSPSA estimates and applies block gradients.
enum class BlockSPSAMode : uint8_t
{
    // Estimate every block at one base point, then apply one full update.
    SnapshotSweepAll,
    // Estimate and update one selected block per iteration.
    RandomOne,
};

constexpr const char* BlockSPSAModeToString(BlockSPSAMode mode)
{
    return mode == BlockSPSAMode::RandomOne ? "random-one" : "snapshot";
}

// Selects how optimizer coordinates are partitioned into blocks.
enum class ParameterBlockStrategy : uint8_t
{
    // Use parameter-aware groups such as gains, matrices, channels, and bands.
    Semantic,
    // Partition the flat coordinate vector into fixed-size contiguous chunks.
    FixedContiguous,
};

constexpr const char* ParameterBlockStrategyToString(ParameterBlockStrategy strategy)
{
    return strategy == ParameterBlockStrategy::FixedContiguous ? "fixed" : "semantic";
}

// Selects how RandomOne mode chooses its next block.
enum class RandomBlockSchedule : uint8_t
{
    // Visit every block once in a seeded random order before reshuffling.
    ShuffledSweep,
    // Sample every block independently with uniform probability.
    IndependentUniform,
};

constexpr const char* RandomBlockScheduleToString(RandomBlockSchedule schedule)
{
    return schedule == RandomBlockSchedule::IndependentUniform ? "uniform" : "shuffled";
}

// Selects how the per-coordinate probe amplitude is scaled by block dimension.
enum class ProbeRadiusNormalization : uint8_t
{
    // Use the same coordinate amplitude in every block.
    None,
    // Divide the coordinate amplitude by sqrt(block size) so every block sweeps the same
    // Euclidean radius. This is a dimension diagnostic, not a theoretically optimal scaling.
    SqrtDimension,
};

constexpr const char* ProbeRadiusNormalizationToString(ProbeRadiusNormalization normalization)
{
    return normalization == ProbeRadiusNormalization::SqrtDimension ? "sqrt-dim" : "none";
}

// Identifies a semantic class of parameter blocks that can share gain scales.
enum class BlockScaleClass : uint8_t
{
    Default,
    GainsInput,
    GainsOutput,
    Matrix,
    Attenuation,
    Tone,
    OverallGain,
};

constexpr const char* BlockScaleClassToString(BlockScaleClass scale_class)
{
    switch (scale_class)
    {
    case BlockScaleClass::GainsInput:
        return "gains_in";
    case BlockScaleClass::GainsOutput:
        return "gains_out";
    case BlockScaleClass::Matrix:
        return "matrix";
    case BlockScaleClass::Attenuation:
        return "attenuation";
    case BlockScaleClass::Tone:
        return "tone";
    case BlockScaleClass::OverallGain:
        return "overall_gain";
    case BlockScaleClass::Default:
    default:
        return "default";
    }
}

// Multiplicative learning-rate and perturbation scales applied to one block class.
struct BlockGainScale
{
    BlockScaleClass scale_class = BlockScaleClass::Default;
    double a_scale = 1.0;
    double c_scale = 1.0;
};

// Parameters for the project-local block-coordinate SPSA optimizer.
struct BlockSPSAParameters
{
    BlockSPSAMode mode = BlockSPSAMode::SnapshotSweepAll;
    ParameterBlockStrategy block_strategy = ParameterBlockStrategy::Semantic;
    RandomBlockSchedule random_schedule = RandomBlockSchedule::ShuffledSweep;
    ThreeBandBlockGrouping three_band_grouping = ThreeBandBlockGrouping::ChannelTriplets;
    ProbeRadiusNormalization probe_radius_normalization = ProbeRadiusNormalization::None;

    size_t contiguous_block_size = 8;
    size_t directions_per_block = 1;

    double alpha = 0.52;
    double gamma = 0.26;
    double step_size = 16.0;
    double evaluation_step_size = 0.42;
    std::optional<double> stability_constant;
    size_t max_iterations = 100000;
    double tolerance = 1e-5;

    // Number of evaluated accepted points over which the best loss must improve by more than
    // `tolerance` (relative) before the run is declared converged. Zero disables early stopping.
    // An unset value derives the window from the block count.
    std::optional<size_t> stall_window;

    // Evaluates the accepted point every N updates instead of every update. Values above one
    // amortize the bookkeeping evaluation that the gradient estimator does not use.
    size_t accepted_evaluation_interval = 1;

    // Caps the Euclidean norm of one coordinate update. Zero disables the cap. Heterogeneous blocks
    // can produce gradients that differ by two orders of magnitude, so an uncapped step can leave the
    // valid parameter region in a single update. Defaults on: the cap is within noise on colorless
    // and prevents divergence on RIR matching, where the tone and overall-gain blocks dominate.
    double max_step_norm = 1.0;

    // Multiplicative per-class gain scales. Later entries override earlier ones for the same class.
    std::vector<BlockGainScale> block_scales;
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
    double step_size = 0.108;
};

enum class OptimizationAlgoType : uint8_t
{
    SPSA,
    BlockSPSA,
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
    case OptimizationAlgoType::BlockSPSA:
        return "BlockSPSA";
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
                 CNEParameters, BlockSPSAParameters>;

struct OptimizationInfo
{
    std::vector<OptimizationParamType> parameters_to_optimize;
    sfFDN::FDNConfig initial_fdn_config;
    uint32_t ir_size;
    fdn_optimization::GradientMethod gradient_method = fdn_optimization::GradientMethod::CentralDifferences;

    std::vector<float> target_rir;
    std::vector<float> early_fir;
    std::vector<float> t60_estimates;
    EarlyFirMode early_fir_mode = EarlyFirMode::DirectPath;

    OptimizationAlgoParams optimizer_params;
    uint32_t seed = 0;
    uint32_t gradient_threads = DefaultGradientThreadCount();
    uint32_t optimizer_threads = 1;
    double max_time_seconds = 0.0;
    uint64_t max_objective_evaluations = 0;
    bool record_trajectory = false;
    MatchingParameterConfig matching_parameters;
};

// Mirrors BlockSPSAProbe for optimizer-agnostic trajectory reporting.
struct BlockSPSAProbeInfo
{
    size_t block = 0;
    size_t block_size = 0;
    size_t visit = 0;
    double learning_rate = 0.0;
    double perturbation = 0.0;
    double probe_plus = 0.0;
    double probe_minus = 0.0;
    double paired_difference = 0.0;
    double absolute_paired_difference = 0.0;
    double gradient_norm = 0.0;
    double step_norm = 0.0;
};

struct OptimizationStepInfo
{
    size_t step = 0;
    double total_loss = 0.0;
    std::vector<double> component_losses;
    double best_loss = 0.0;
    double learning_rate = 0.0;
    double gradient_norm = 0.0;
    uint64_t objective_evaluations = 0;
    std::chrono::duration<double> elapsed_time;
    std::optional<size_t> active_block;
    std::optional<size_t> block_visit;
    std::optional<double> perturbation;
    std::optional<size_t> directions_averaged;
    std::optional<bool> evaluated;
    std::optional<bool> improved_best;
    std::optional<double> step_norm;
    std::vector<BlockSPSAProbeInfo> block_probes;
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
    std::chrono::duration<double> total_time{};
    std::chrono::duration<double> setup_time{};
    std::chrono::duration<double> initial_evaluation_time{};
    std::chrono::duration<double> optimizer_time{};
    std::chrono::duration<double> final_evaluation_time{};
    uint32_t total_evaluations = 0;
    uint64_t objective_evaluations = 0;
    uint32_t gradient_threads = 1;
    uint32_t optimizer_threads = 1;
    std::vector<std::vector<double>> loss_history;
    std::vector<std::string> loss_names;
    std::vector<double> final_losses;
    std::vector<OptimizationStepInfo> trajectory;
    double best_loss = 0.0;
    std::string termination_reason;
    std::optional<size_t> parameter_block_count;
    std::optional<double> optimizer_stability_constant;
    std::optional<size_t> optimizer_stall_window;
};

class OptimCallback;

class FDNOptimizer
{
  public:
    FDNOptimizer(quill::Logger* logger, bool verbose = false);
    ~FDNOptimizer();

    FDNOptimizer(const FDNOptimizer&) = delete;
    FDNOptimizer& operator=(const FDNOptimizer&) = delete;

    FDNOptimizer(FDNOptimizer&&) = delete;
    FDNOptimizer& operator=(FDNOptimizer&&) = delete;

    void SetLossFunctions(std::span<std::shared_ptr<AudioLoss>> loss_functions);

    void StartOptimization(OptimizationInfo& info);
    void CancelOptimization();
    void ResetStatus();

    OptimizationStatus GetStatus() const;
    OptimizationStatus WaitForCompletion();
    OptimizationProgressInfo GetProgress();

    OptimizationResult GetResult();

  private:
    void ThreadProc(std::stop_token stop_token, OptimizationInfo info);
    void SetStatus(OptimizationStatus status);

    quill::Logger* logger_;
    bool verbose_;
    std::atomic<OptimizationStatus> status_;

    std::chrono::steady_clock::time_point start_time_;
    std::jthread thread_;
    std::mutex mutex_;
    std::condition_variable status_cv_;

    sfFDN::FDNConfig optimized_config_;
    OptimizationResult optimization_result_;

    std::unique_ptr<OptimCallback> optim_callback_;

    std::vector<std::shared_ptr<AudioLoss>> loss_functions_;
};
} // namespace fdn_optimization