#pragma once

#include <sffdn/sffdn.h>

#include <audio_utils/audio_file_manager.h>

#include <quill/LogMacros.h>
#include <quill/Logger.h>

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

    // Format is
    // Row 1: input_gains
    // Row 2: output_gains
    // Row 3: delays
    // Row 4: t60s
    // Row 5: tone correction gains
    // Next N rows: feedback matrix
    for (const auto& gain : config.input_gains)
    {
        file << gain << " ";
    }
    file << std::endl;

    for (const auto& gain : config.output_gains)
    {
        file << gain << " ";
    }
    file << std::endl;

    for (const auto& delay : config.delays)
    {
        file << delay << " ";
    }
    file << std::endl;

    if (std::holds_alternative<std::vector<float>>(config.matrix_info))
    {
        const auto& matrix_coeffs = std::get<std::vector<float>>(config.matrix_info);
        const uint32_t N = config.N;
        for (uint32_t r = 0; r < N; ++r)
        {
            for (uint32_t c = 0; c < N; ++c)
            {
                file << matrix_coeffs[r * N + c] << " ";
            }
            file << std::endl;
        }
    }
    else
    {
        LOG_ERROR(logger, "Feedback matrix is not in expected format for writing to file.");
    }
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

    // Format is
    // Row 1: t60s
    // Row 2: tone correction frequencies
    // Row 3: tone correction gains

    for (const auto& t60 : config.attenuation_t60s)
    {
        file << t60 << " ";
    }
    file << std::endl;

    for (const auto& freq : config.tc_frequencies)
    {
        file << freq << " ";
    }
    file << std::endl;

    for (const auto& gain : config.tc_gains)
    {
        file << gain << " ";
    }
    file << std::endl;
}

inline void SaveImpulseResponse(const sfFDN::FDNConfig& config, uint32_t ir_length,
                                const std::filesystem::path& filename, quill::Logger* logger)
{
    auto config_copy = config;
    // config_copy.attenuation_t60s = {1.f};

    auto fdn = sfFDN::CreateFDNFromConfig(config_copy, kSampleRate);
    fdn->SetDirectGain(0.0f);

    std::vector<float> input_data(ir_length, 0.0f);
    input_data[0] = 1.0f; // Delta impulse

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