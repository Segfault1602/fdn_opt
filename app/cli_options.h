#pragma once

#include "optimizer.h"

#include <CLI/CLI.hpp>

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fdn_opt_app
{

struct ExecutionOptions
{
    fdn_optimization::GradientMethod gradient_method = fdn_optimization::GradientMethod::CentralDifferences;
    uint32_t seed = 0;
    uint32_t gradient_threads = fdn_optimization::DefaultGradientThreadCount();
    uint32_t optimizer_threads = 1;
    double max_time_seconds = 0.0;
    uint64_t max_objective_evaluations = 0;
    size_t sparsity_window_samples = 4096;
    uint32_t spectral_fft_size = 65536;
    bool record_trajectory = false;
};

enum class MatchingFilterType
{
    TenBand,
    ThreeBand,
};

struct MatchingAnalysisOptions
{
    MatchingFilterType filter_type = MatchingFilterType::TenBand;
    fdn_optimization::EarlyFirMode early_fir_mode = fdn_optimization::EarlyFirMode::DirectPath;
    uint32_t fft_length = 4096;
    uint32_t hop_size = 128;
    uint32_t window_size = 1024;
    uint32_t mel_bands = 32;
    fdn_optimization::MatchingParameterConfig parameter_config;
};

struct InputOptions
{
    std::string ir_filename;
    std::string early_fir_path;
    uint32_t fdn_order = 6;
    bool colorless_only = false;
    bool matching_only = false;
    std::string colorless_config_path;
};

struct LossOptions
{
    double spectral_flatness_weight = 1.0;
    double sparsity_weight = 0.5;
    double power_envelope_weight = 0.0;
    double edc_weight = 0.1;
    double mel_edr_weight = 1.0;
    double weighted_edr_weight = 0.0;
};

struct InitializationOptions
{
    bool randomize_initial = false;
    bool random_delays = false;
    // Seeds the initial FDN configuration. Unset reuses the execution seed, which conflates the
    // problem instance with optimizer stochasticity and prevents separating robustness from luck.
    std::optional<uint32_t> init_seed;
};

struct OutputOptions
{
    bool save_output = true;
    bool no_save_output = false;
    std::string output_dir = "optim_output";
    std::string result_json_path;
    std::string spectrum_result_json_path;
    std::string trajectory_jsonl_path;
    std::string spectrum_trajectory_jsonl_path;
};

struct OptimizerCliOptions
{
    fdn_optimization::OptimizationAlgoParams selected_params;
    std::string selected_name;
    fdn_optimization::AdamParameters adam;
    fdn_optimization::SPSAParameters spsa;
    fdn_optimization::BlockSPSAParameters block_spsa;
    // Unparsed `<class>:<a_scale>:<c_scale>` specifications for BlockSPSA block gain scales.
    std::vector<std::string> block_spsa_scale_specs;
    fdn_optimization::SimulatedAnnealingParameters simulated_annealing;
    fdn_optimization::CNEParameters cne;
    fdn_optimization::DifferentialEvolutionParameters differential_evolution;
    fdn_optimization::PSOParameters pso;
    fdn_optimization::L_BFGSParameters lbfgs;
    fdn_optimization::GradientDescentParameters gradient_descent;
    fdn_optimization::CMAESParameters cmaes;
    fdn_optimization::RandomSearchParameters random_search;
};

struct ParsedCliOptions
{
    InputOptions input;
    LossOptions losses;
    InitializationOptions initialization;
    OutputOptions output;
    ExecutionOptions colorless_execution;
    ExecutionOptions matching_execution;
    MatchingAnalysisOptions matching;
    fdn_optimization::OptimizationAlgoParams optimizer_params;
    std::string selected_optimizer;
    std::string config_filename;
    bool save_output = true;
    bool verbose = false;
};

struct RawCliOptions
{
    InputOptions input;
    LossOptions losses;
    InitializationOptions initialization;
    OutputOptions output;
    ExecutionOptions execution;
    MatchingAnalysisOptions matching;
    OptimizerCliOptions optimizer;
    bool verbose = false;
    std::optional<ParsedCliOptions> parsed;
};

void ConfigureCliApp(CLI::App& app, RawCliOptions& options);
std::expected<ParsedCliOptions, std::string> ValidateAndNormalizeCliOptions(const RawCliOptions& options);

std::string_view GradientMethodName(fdn_optimization::GradientMethod method);
std::string_view MatchingFilterTypeName(MatchingFilterType type);
std::string_view EarlyFirModeName(fdn_optimization::EarlyFirMode mode);
std::string_view MatchingParameterizationName(fdn_optimization::MatchingParameterization parameterization);
std::string_view MatchingInitializationName(fdn_optimization::MatchingInitialization initialization);

} // namespace fdn_opt_app
