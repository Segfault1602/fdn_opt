#pragma once

#include <sffdn/sffdn.h>

#include <audio_utils/audio_file_manager.h>

#include <nlohmann/json.hpp>
#include <quill/LogMacros.h>
#include <quill/Logger.h>

#include <armadillo>

#include <filesystem>
#include <fstream>

constexpr uint32_t kSampleRate = 48000;

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

    std::visit(
        [&](const auto& params) {
            using T = std::decay_t<decltype(params)>;
            if constexpr (std::is_same_v<T, fdn_optimization::AdamParameters>)
            {
                file << "Optimizer: Adam" << std::endl;
                file << "    Step Size: " << params.step_size << std::endl;
                file << "    Beta1: " << params.beta1 << std::endl;
                file << "    Beta2: " << params.beta2 << std::endl;
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

inline void SaveImpulseResponse(const sfFDN::FDNConfig& config, uint32_t ir_length,
                                const std::filesystem::path& filename, quill::Logger* logger,
                                const std::vector<float>& early_fir = {})
{
    auto config_copy = config;
    // config_copy.attenuation_t60s = {1.f};

    auto fdn = sfFDN::CreateFDNFromConfig(config_copy);
    fdn->SetDirectGain(0.0f);

    std::vector<float> input_data(ir_length, 0.0f);

    if (early_fir.empty())
    {
        input_data[0] = 1.0f; // Delta impulse
    }
    else
    {
        std::copy(early_fir.begin(), early_fir.end(), input_data.begin());
    }

    std::vector<float> impulse_response(ir_length, 0.0f);
    sfFDN::AudioBuffer impulse_buffer(impulse_response);

    sfFDN::AudioBuffer in_buffer(input_data);
    fdn->Process(in_buffer, impulse_buffer);

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