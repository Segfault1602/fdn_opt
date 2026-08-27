#pragma once

#include "optimizer.h"

#include <sffdn/sffdn.h>

#include <audio_utils/audio_file_manager.h>
#include <audio_utils/fft.h>

#include <nlohmann/json.hpp>
#include <quill/LogMacros.h>
#include <quill/Logger.h>

#include <armadillo>

#include <algorithm>
#include <filesystem>
#include <fstream>

constexpr uint32_t kSampleRate = 48000;

inline nlohmann::json OptimizationParametersToJson(const fdn_optimization::OptimizationAlgoParams& optimizer_params)
{
    return std::visit(
        [](const auto& params) {
            using T = std::decay_t<decltype(params)>;
            nlohmann::json json;
            if constexpr (std::is_same_v<T, fdn_optimization::AdamParameters>)
            {
                json = {{"step_size", params.step_size},
                        {"beta1", params.beta1},
                        {"beta2", params.beta2},
                        {"learning_rate_decay", params.learning_rate_decay},
                        {"decay_step_size", params.decay_step_size},
                        {"epoch_restarts", params.epoch_restarts},
                        {"max_restarts", params.max_restarts},
                        {"max_iterations", params.max_iterations},
                        {"tolerance", params.tolerance},
                        {"gradient_delta", params.gradient_delta}};
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::L_BFGSParameters>)
            {
                json = {{"num_basis", params.num_basis},
                        {"max_iterations", params.max_iterations},
                        {"wolfe", params.wolfe},
                        {"min_gradient_norm", params.min_gradient_norm},
                        {"factor", params.factor},
                        {"max_line_search_trials", params.max_line_search_trials},
                        {"min_step", params.min_step},
                        {"max_step", params.max_step},
                        {"gradient_delta", params.gradient_delta}};
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::GradientDescentParameters>)
            {
                json = {{"step_size", params.step_size},
                        {"max_iterations", params.max_iterations},
                        {"tolerance", params.tolerance},
                        {"kappa", params.kappa},
                        {"phi", params.phi},
                        {"momentum", params.momentum},
                        {"min_gain", params.min_gain},
                        {"gradient_delta", params.gradient_delta},
                        {"max_step_norm", params.max_step_norm}};
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::SPSAParameters>)
            {
                json = {{"alpha", params.alpha},
                        {"gamma", params.gamma},
                        {"step_size", params.step_size},
                        {"evaluation_step_size", params.evaluationStepSize},
                        {"max_iterations", params.max_iterations},
                        {"tolerance", params.tolerance}};
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::BlockSPSAParameters>)
            {
                json = {{"mode", fdn_optimization::BlockSPSAModeToString(params.mode)},
                        {"block_strategy", fdn_optimization::ParameterBlockStrategyToString(params.block_strategy)},
                        {"random_schedule", fdn_optimization::RandomBlockScheduleToString(params.random_schedule)},
                        {"three_band_grouping",
                         fdn_optimization::ThreeBandBlockGroupingToString(params.three_band_grouping)},
                        {"block_size", params.contiguous_block_size},
                        {"directions_per_block", params.directions_per_block},
                        {"alpha", params.alpha},
                        {"gamma", params.gamma},
                        {"step_size", params.step_size},
                        {"evaluation_step_size", params.evaluation_step_size},
                        {"stability_constant", params.stability_constant},
                        {"stall_window", params.stall_window},
                        {"probe_radius_normalization",
                         fdn_optimization::ProbeRadiusNormalizationToString(params.probe_radius_normalization)},
                        {"accepted_evaluation_interval", params.accepted_evaluation_interval},
                        {"max_step_norm", params.max_step_norm},
                        {"max_iterations", params.max_iterations},
                        {"tolerance", params.tolerance}};
                nlohmann::json block_scales = nlohmann::json::array();
                for (const auto& scale : params.block_scales)
                {
                    block_scales.push_back({{"class", fdn_optimization::BlockScaleClassToString(scale.scale_class)},
                                            {"a_scale", scale.a_scale},
                                            {"c_scale", scale.c_scale}});
                }
                json["block_scales"] = std::move(block_scales);
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::SimulatedAnnealingParameters>)
            {
                json = {{"max_iterations", params.max_iterations},
                        {"initial_temperature", params.initial_temperature},
                        {"init_moves", params.init_moves},
                        {"move_ctrl_sweep", params.move_ctrl_sweep},
                        {"max_tolerance_sweep", params.max_tolerance_sweep},
                        {"max_move_coefficient", params.max_move_coef},
                        {"init_move_coefficient", params.init_move_coef},
                        {"gain", params.gain},
                        {"tolerance", params.tolerance}};
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::CNEParameters>)
            {
                json = {{"population_size", params.population_size},
                        {"max_generations", params.max_generations},
                        {"mutation_probability", params.mutation_probability},
                        {"mutation_size", params.mutation_size},
                        {"select_percent", params.select_percent},
                        {"tolerance", params.tolerance}};
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::DifferentialEvolutionParameters>)
            {
                json = {{"population_size", params.population_size},
                        {"max_generations", params.max_generation},
                        {"crossover_rate", params.crossover_rate},
                        {"differential_weight", params.differential_weight},
                        {"tolerance", params.tolerance}};
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::PSOParameters>)
            {
                json = {{"num_particles", params.num_particles},
                        {"max_iterations", params.max_iterations},
                        {"horizon_size", params.horizon_size},
                        {"exploitation_factor", params.exploitation_factor},
                        {"exploration_factor", params.exploration_factor},
                        {"tolerance", params.tolerance}};
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::CMAESParameters>)
            {
                json = {{"population_size", params.population_size},
                        {"max_iterations", params.max_iterations},
                        {"tolerance", params.tolerance},
                        {"step_size", params.step_size}};
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::RandomSearchParameters>)
            {
                json = {{"time_limit_seconds", params.time_limit_seconds}};
            }
            return json;
        },
        optimizer_params);
}

inline nlohmann::json OptimizationResultToJson(const fdn_optimization::OptimizationResult& result,
                                               const fdn_optimization::OptimizationAlgoParams& optimizer_params,
                                               const nlohmann::json& run_metadata)
{
    nlohmann::json loss_components = nlohmann::json::object();
    for (size_t i = 0; i < result.loss_names.size() && i < result.final_losses.size(); ++i)
        loss_components[result.loss_names[i]] = result.final_losses[i];

    nlohmann::json initial_loss_components = nlohmann::json::object();
    for (size_t i = 0; i < result.loss_names.size() && i < result.initial_losses.size(); ++i)
        initial_loss_components[result.loss_names[i]] = result.initial_losses[i];

    const auto& config = result.optimized_fdn_config;
    arma::fvec input_gains(config.input_block_config.parallel_gains_config.gains);
    arma::fvec output_gains(config.output_block_config.parallel_gains_config.gains);

    double orthogonality_error = 0.0;
    if (std::holds_alternative<sfFDN::ScalarFeedbackMatrixOptions>(config.feedback_matrix_config))
    {
        const auto& options = std::get<sfFDN::ScalarFeedbackMatrixOptions>(config.feedback_matrix_config);
        sfFDN::ScalarFeedbackMatrix matrix(options);
        arma::mat coefficients(config.fdn_size, config.fdn_size);
        for (uint32_t row = 0; row < config.fdn_size; ++row)
        {
            for (uint32_t col = 0; col < config.fdn_size; ++col)
                coefficients(row, col) = matrix.GetCoefficient(row, col);
        }
        orthogonality_error =
            arma::norm(coefficients.t() * coefficients - arma::eye(config.fdn_size, config.fdn_size), "fro");
    }

    nlohmann::json json = {
        {"execution", run_metadata},
        {"optimizer_parameters", OptimizationParametersToJson(optimizer_params)},
        {"timing",
         {{"total_seconds", result.total_time.count()},
          {"setup_seconds", result.setup_time.count()},
          {"initial_evaluation_seconds", result.initial_evaluation_time.count()},
          {"optimizer_seconds", result.optimizer_time.count()},
          {"final_evaluation_seconds", result.final_evaluation_time.count()}}},
        {"optimizer_callbacks", result.total_evaluations},
        {"objective_evaluations", result.objective_evaluations},
        {"effective_gradient_threads", result.gradient_threads},
        {"effective_optimizer_threads", result.optimizer_threads},
        {"termination_reason", result.termination_reason},
        {"initial_loss", result.initial_loss},
        {"initial_loss_components", initial_loss_components},
        {"final_loss", result.best_loss},
        {"loss_components", loss_components},
        {"initial_fdn_config", result.initial_fdn_config},
        {"optimized_fdn_config", result.optimized_fdn_config},
        {"validity",
         {{"input_gain_norm", arma::norm(input_gains, 2)},
          {"output_gain_norm", arma::norm(output_gains, 2)},
          {"matrix_orthogonality_frobenius_error", orthogonality_error}}},
    };
    if (result.parameter_block_count)
    {
        json["parameter_block_count"] = *result.parameter_block_count;
    }
    if (result.optimizer_stall_window)
    {
        json["optimizer_stall_window"] = *result.optimizer_stall_window;
    }
    if (result.optimizer_stability_constant)
    {
        json["optimizer_stability_constant"] = *result.optimizer_stability_constant;
    }
    return json;
}

inline bool WriteJsonToFile(const nlohmann::json& json, const std::filesystem::path& filename, quill::Logger* logger)
{
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open())
    {
        LOG_ERROR(logger, "Failed to open file {} for writing JSON.", filename.string());
        return false;
    }
    file << json.dump(2) << '\n';
    file.flush();
    return file.good();
}

inline bool WriteTrajectoryJsonl(const std::vector<fdn_optimization::OptimizationStepInfo>& trajectory,
                                 const std::vector<std::string>& loss_names, const std::filesystem::path& filename,
                                 quill::Logger* logger)
{
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open())
    {
        LOG_ERROR(logger, "Failed to open file {} for writing trajectory.", filename.string());
        return false;
    }

    for (const auto& step : trajectory)
    {
        nlohmann::json component_losses = nlohmann::json::object();
        for (size_t i = 0; i < loss_names.size() && i < step.component_losses.size(); ++i)
            component_losses[loss_names[i]] = step.component_losses[i];

        nlohmann::json row = {{"step", step.step},
                              {"total_loss", step.total_loss},
                              {"component_losses", component_losses},
                              {"best_loss", step.best_loss},
                              {"learning_rate", step.learning_rate},
                              {"gradient_norm", step.gradient_norm},
                              {"objective_evaluations", step.objective_evaluations},
                              {"elapsed_seconds", step.elapsed_time.count()}};
        if (step.active_block)
        {
            row["active_block"] = *step.active_block;
        }
        if (step.block_visit)
        {
            row["block_visit"] = *step.block_visit;
        }
        if (step.perturbation)
        {
            row["perturbation"] = *step.perturbation;
        }
        if (step.directions_averaged)
        {
            row["directions_averaged"] = *step.directions_averaged;
        }
        if (step.evaluated)
        {
            row["evaluated"] = *step.evaluated;
        }
        if (step.improved_best)
        {
            row["improved_best"] = *step.improved_best;
        }
        if (step.step_norm)
        {
            row["step_norm"] = *step.step_norm;
        }
        if (!step.block_probes.empty())
        {
            nlohmann::json probes = nlohmann::json::array();
            for (const auto& probe : step.block_probes)
            {
                probes.push_back({{"block", probe.block},
                                  {"block_size", probe.block_size},
                                  {"visit", probe.visit},
                                  {"learning_rate", probe.learning_rate},
                                  {"perturbation", probe.perturbation},
                                  {"probe_plus", probe.probe_plus},
                                  {"probe_minus", probe.probe_minus},
                                  {"paired_difference", probe.paired_difference},
                                  {"absolute_paired_difference", probe.absolute_paired_difference},
                                  {"gradient_norm", probe.gradient_norm},
                                  {"step_norm", probe.step_norm}});
            }
            row["block_probes"] = std::move(probes);
        }
        file << row.dump() << '\n';
    }
    file.flush();
    return file.good();
}

inline void WriteConfigToFile(const sfFDN::FDNConfig& config, const std::filesystem::path& filename,
                              quill::Logger* logger)
{
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open())
    {
        LOG_ERROR(logger, "Failed to open file {} for writing FDNConfig.", filename.string());
        return;
    }

    nlohmann::json j = config;
    file << j.dump(4);
}

inline std::optional<sfFDN::FDNConfig> ReadConfigFromFile(const std::filesystem::path& filename, quill::Logger* logger)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        LOG_ERROR(logger, "Failed to open file {} for reading FDNConfig.", filename.string());
        return std::nullopt;
    }

    try
    {
        nlohmann::json json;
        file >> json;
        return json.get<sfFDN::FDNConfig>();
    }
    catch (const std::exception& error)
    {
        LOG_ERROR(logger, "Failed to parse FDNConfig from {}: {}", filename.string(), error.what());
        return std::nullopt;
    }
}

inline void WriteInfoToFile(const fdn_optimization::OptimizationResult& result,
                            const fdn_optimization::OptimizationAlgoParams& optimizer_params,
                            const std::filesystem::path& filename, quill::Logger* logger)
{
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open())
    {
        LOG_ERROR(logger, "Failed to open file {} for writing optimization info.", filename.string());
        return;
    }

    file << "Best Loss: " << result.best_loss << std::endl;
    file << "Total Time (s): " << result.total_time.count() << std::endl;
    file << "Total Evaluations: " << result.total_evaluations << std::endl;
    file << "Objective Evaluations: " << result.objective_evaluations << std::endl;
    file << "Termination Reason: " << result.termination_reason << std::endl;

    std::visit(
        [&](const auto& params) {
            using T = std::decay_t<decltype(params)>;
            if constexpr (std::is_same_v<T, fdn_optimization::AdamParameters>)
            {
                file << "Optimizer: Adam" << std::endl;
                file << "    Step Size: " << params.step_size << std::endl;
                file << "    Beta1: " << params.beta1 << std::endl;
                file << "    Beta2: " << params.beta2 << std::endl;
                file << "    Max Iterations: " << params.max_iterations << std::endl;
                file << "    Tolerance: " << params.tolerance << std::endl;
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::L_BFGSParameters>)
            {
                file << "Optimizer: L-BFGS" << std::endl;
                file << "    Num Basis: " << params.num_basis << std::endl;
                file << "    Max Iterations: " << params.max_iterations << std::endl;
                file << "    Wolfe: " << params.wolfe << std::endl;
                file << "    Min Gradient Norm: " << params.min_gradient_norm << std::endl;
                file << "    Factor: " << params.factor << std::endl;
                file << "    Max Line Search Trials: " << params.max_line_search_trials << std::endl;
                file << "    Min Step: " << params.min_step << std::endl;
                file << "    Max Step: " << params.max_step << std::endl;
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::SPSAParameters>)
            {
                file << "Optimizer: SPSA" << std::endl;
                file << "    Alpha: " << params.alpha << std::endl;
                file << "    Gamma: " << params.gamma << std::endl;
                file << "    Step Size: " << params.step_size << std::endl;
                file << "    Evaluation Step Size: " << params.evaluationStepSize << std::endl;
                file << "    Max Iterations: " << params.max_iterations << std::endl;
                file << "    Tolerance: " << params.tolerance << std::endl;
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::BlockSPSAParameters>)
            {
                file << "Optimizer: BlockSPSA" << std::endl;
                file << "    Mode: " << fdn_optimization::BlockSPSAModeToString(params.mode) << std::endl;
                file << "    Block Strategy: "
                     << fdn_optimization::ParameterBlockStrategyToString(params.block_strategy) << std::endl;
                file << "    Random Schedule: " << fdn_optimization::RandomBlockScheduleToString(params.random_schedule)
                     << std::endl;
                file << "    Three-Band Grouping: "
                     << fdn_optimization::ThreeBandBlockGroupingToString(params.three_band_grouping) << std::endl;
                file << "    Block Size: " << params.contiguous_block_size << std::endl;
                file << "    Directions Per Block: " << params.directions_per_block << std::endl;
                file << "    Alpha: " << params.alpha << std::endl;
                file << "    Gamma: " << params.gamma << std::endl;
                file << "    Step Size: " << params.step_size << std::endl;
                file << "    Evaluation Step Size: " << params.evaluation_step_size << std::endl;
                file << "    Stability Constant: "
                     << (params.stability_constant ? std::to_string(*params.stability_constant)
                                                   : std::string("default"))
                     << std::endl;
                file << "    Stall Window: "
                     << (params.stall_window ? std::to_string(*params.stall_window) : std::string("default"))
                     << std::endl;
                file << "    Probe Radius Normalization: "
                     << fdn_optimization::ProbeRadiusNormalizationToString(params.probe_radius_normalization)
                     << std::endl;
                file << "    Accepted Evaluation Interval: " << params.accepted_evaluation_interval << std::endl;
                file << "    Max Step Norm: " << params.max_step_norm << std::endl;
                for (const auto& scale : params.block_scales)
                {
                    file << "    Block Scale [" << fdn_optimization::BlockScaleClassToString(scale.scale_class)
                         << "]: a=" << scale.a_scale << " c=" << scale.c_scale << std::endl;
                }
                file << "    Max Iterations: " << params.max_iterations << std::endl;
                file << "    Tolerance: " << params.tolerance << std::endl;
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::SimulatedAnnealingParameters>)
            {
                file << "Optimizer: Simulated Annealing" << std::endl;
                file << "    Max Iterations: " << params.max_iterations << std::endl;
                file << "    Initial Temperature: " << params.initial_temperature << std::endl;
                file << "    Init Moves: " << params.init_moves << std::endl;
                file << "    Move Ctrl Sweep: " << params.move_ctrl_sweep << std::endl;
                file << "    Max Tolerance Sweep: " << params.max_tolerance_sweep << std::endl;
                file << "    Max Move Coef: " << params.max_move_coef << std::endl;
                file << "    Init Move Coef: " << params.init_move_coef << std::endl;
                file << "    Gain: " << params.gain << std::endl;
                file << "    Tolerance: " << params.tolerance << std::endl;
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::GradientDescentParameters>)
            {
                file << "Optimizer: Gradient Descent" << std::endl;
                file << "    Step Size: " << params.step_size << std::endl;
                file << "    Max Iterations: " << params.max_iterations << std::endl;
                file << "    Tolerance: " << params.tolerance << std::endl;
                file << "    Kappa: " << params.kappa << std::endl;
                file << "    Phi: " << params.phi << std::endl;
                file << "    Momentum: " << params.momentum << std::endl;
                file << "    Min Gain: " << params.min_gain << std::endl;
                file << "    Gradient Delta: " << params.gradient_delta << std::endl;
                file << "    Max Step Norm: " << params.max_step_norm << std::endl;
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::CMAESParameters>)
            {
                file << "Optimizer: CMA-ES" << std::endl;
                file << "    Population Size: " << params.population_size << std::endl;
                file << "    Max Iterations: " << params.max_iterations << std::endl;
                file << "    Tolerance: " << params.tolerance << std::endl;
                file << "    Step Size: " << params.step_size << std::endl;
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::DifferentialEvolutionParameters>)
            {
                file << "Optimizer: Differential Evolution" << std::endl;
                file << "    Population Size: " << params.population_size << std::endl;
                file << "    Max Generation: " << params.max_generation << std::endl;
                file << "    Crossover Rate: " << params.crossover_rate << std::endl;
                file << "    Differential Weight: " << params.differential_weight << std::endl;
                file << "    Tolerance: " << params.tolerance << std::endl;
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::PSOParameters>)
            {
                file << "Optimizer: PSO" << std::endl;
                file << "    Num Particles: " << params.num_particles << std::endl;
                file << "    Max Iterations: " << params.max_iterations << std::endl;
                file << "    Horizon Size: " << params.horizon_size << std::endl;
                file << "    Exploitation Factor: " << params.exploitation_factor << std::endl;
                file << "    Exploration Factor: " << params.exploration_factor << std::endl;
                file << "    Tolerance: " << params.tolerance << std::endl;
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::CNEParameters>)
            {
                file << "Optimizer: CNE" << std::endl;
                file << "    Population Size: " << params.population_size << std::endl;
                file << "    Max Generations: " << params.max_generations << std::endl;
                file << "    Mutation Probability: " << params.mutation_probability << std::endl;
                file << "    Mutation Size: " << params.mutation_size << std::endl;
                file << "    Select Percent: " << params.select_percent << std::endl;
                file << "    Tolerance: " << params.tolerance << std::endl;
            }
            else if constexpr (std::is_same_v<T, fdn_optimization::RandomSearchParameters>)
            {
                file << "Optimizer: Random Search" << std::endl;
                file << "    Time Limit (s): " << params.time_limit_seconds << std::endl;
            }
            else
            {
                file << "Optimizer: Unknown" << std::endl;
            }
        },
        optimizer_params);
}

inline void WriteFilterConfigToFile(const sfFDN::FDNConfig& config, const std::filesystem::path& filename,
                                    quill::Logger* logger)
{
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open())
    {
        LOG_ERROR(logger, "Failed to open file {} for writing FDNConfig.", filename.string());
        return;
    }

    nlohmann::json j = config;
    file << j.dump(4); // Pretty print with 4 spaces indentation
}

inline void SaveImpulseResponse(
    const sfFDN::FDNConfig& config, uint32_t ir_length, const std::filesystem::path& filename, quill::Logger* logger,
    const std::vector<float>& early_fir = {},
    fdn_optimization::EarlyFirMode early_fir_mode = fdn_optimization::EarlyFirMode::DirectPath)
{
    auto config_copy = config;
    // config_copy.attenuation_t60s = {1.f};

    auto fdn = sfFDN::CreateFDNFromConfig(config_copy);
    fdn->SetDirectGain(0.0f);

    std::vector<float> input_data(ir_length, 0.0f);

    if (early_fir.empty() || early_fir_mode == fdn_optimization::EarlyFirMode::DirectPath)
        input_data[0] = 1.0f;
    else
        std::copy_n(early_fir.begin(), std::min(input_data.size(), early_fir.size()), input_data.begin());

    std::vector<float> impulse_response(ir_length, 0.0f);
    sfFDN::AudioBuffer impulse_buffer(impulse_response);

    sfFDN::AudioBuffer in_buffer(input_data);
    fdn->Process(in_buffer, impulse_buffer);

    if (!early_fir.empty() && early_fir_mode == fdn_optimization::EarlyFirMode::DirectPath)
    {
        const size_t copy_size = std::min(impulse_response.size(), early_fir.size());
        for (size_t index = 0; index < copy_size; ++index)
            impulse_response[index] += early_fir[index];
    }

    LOG_INFO(logger, "Writing impulse response to file: {}", filename.string());
    audio_utils::audio_file::WriteWavFile(filename.string(), impulse_response, kSampleRate);
}

inline void WriteLossHistoryToFile(const std::vector<std::vector<double>>& loss_history,
                                   const std::vector<std::string>& loss_names, const std::filesystem::path& filename,
                                   quill::Logger* logger)
{
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open())
    {
        LOG_ERROR(logger, "Failed to open file {} for writing loss history.", filename.string());
        return;
    }

    // Check that all loss vectors have the same length
    size_t history_length = loss_history[0].size();
    for (const auto& losses : loss_history)
    {
        if (losses.size() != history_length)
        {
            LOG_ERROR(logger, "Inconsistent loss history lengths when writing to file {}.", filename.string());
            return;
        }
    }

    // Write header
    file << "Total, ";
    for (size_t i = 0; i < loss_names.size(); ++i)
    {
        file << loss_names[i];
        if (i < loss_names.size() - 1)
        {
            file << ", ";
        }
    }
    file << std::endl;

    for (size_t i = 0; i < history_length; ++i)
    {
        for (size_t j = 0; j < loss_history.size(); ++j)
        {
            file << loss_history[j][i];
            if (j < loss_history.size() - 1)
            {
                file << ", ";
            }
        }
        file << std::endl;
    }
}