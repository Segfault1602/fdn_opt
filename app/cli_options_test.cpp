#include "cli_options.h"

#include <CLI/CLI.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

namespace
{

fdn_opt_app::ParsedCliOptions Parse(const std::string& command_line)
{
    CLI::App app{"test"};
    fdn_opt_app::RawCliOptions raw;
    fdn_opt_app::ConfigureCliApp(app, raw);
    app.parse(command_line, false);
    if (!raw.parsed)
    {
        throw std::runtime_error("CLI final callback did not produce parsed options.");
    }
    return *raw.parsed;
}

fdn_opt_app::RawCliOptions ValidRawOptions()
{
    fdn_opt_app::RawCliOptions raw;
    raw.input.colorless_only = true;
    raw.optimizer.selected_name = "Adam";
    raw.optimizer.selected_params = raw.optimizer.adam;
    return raw;
}

bool ParseFails(const std::string& command_line)
{
    try
    {
        [[maybe_unused]] auto parsed = Parse(command_line);
        return false;
    }
    catch (const CLI::ParseError&)
    {
        return true;
    }
}

} // namespace

int main()
{
    const auto defaults = Parse("--colorless_only Adam");
    if (defaults.input.fdn_order != 6 ||
        defaults.colorless_execution.gradient_method != fdn_optimization::GradientMethod::CentralDifferences ||
        defaults.matching.filter_type != fdn_opt_app::MatchingFilterType::TenBand ||
        defaults.matching.early_fir_mode != fdn_optimization::EarlyFirMode::DirectPath ||
        defaults.matching.parameter_config.parameterization !=
            fdn_optimization::MatchingParameterization::ScaledSmooth ||
        defaults.matching.parameter_config.initialization != fdn_optimization::MatchingInitialization::Neutral ||
        defaults.selected_optimizer != "Adam" || !defaults.save_output)
    {
        return 1;
    }

    const auto transformed =
        Parse("--colorless_only --gradient_method forward --matching_filter_type 3band "
              "--early_fir_mode excitation --matching_parameterization raw --matching_initialization target "
              "--no-save-output --trajectory_jsonl colorless.jsonl "
              "--spectrum_trajectory_jsonl matching.jsonl Adam --step_size 0.25 --max_iterations 12");
    if (transformed.colorless_execution.gradient_method != fdn_optimization::GradientMethod::ForwardDifferences ||
        transformed.matching.filter_type != fdn_opt_app::MatchingFilterType::ThreeBand ||
        transformed.matching.early_fir_mode != fdn_optimization::EarlyFirMode::Excitation ||
        transformed.matching.parameter_config.parameterization !=
            fdn_optimization::MatchingParameterization::RawClamped ||
        transformed.matching.parameter_config.initialization !=
            fdn_optimization::MatchingInitialization::TargetDerived ||
        transformed.save_output || !transformed.colorless_execution.record_trajectory ||
        !transformed.matching_execution.record_trajectory)
    {
        return 2;
    }
    const auto& transformed_adam = std::get<fdn_optimization::AdamParameters>(transformed.optimizer_params);
    if (std::abs(transformed_adam.step_size - 0.25f) > 1e-7f || transformed_adam.max_iterations != 12)
    {
        return 3;
    }

    const auto random_initialization = Parse("--colorless_only --matching_initialization random Adam");
    if (random_initialization.matching.parameter_config.initialization !=
        fdn_optimization::MatchingInitialization::SeededRandom)
    {
        return 4;
    }

    for (const std::string optimizer : {"SPSA", "SimulatedAnnealing", "CNE", "DifferentialEvolution", "PSO", "L-BFGS",
                                        "GradientDescent", "CMAES", "RandomSearch"})
    {
        if (Parse("--colorless_only " + optimizer).selected_optimizer != optimizer)
        {
            return 5;
        }
    }

    if (!ParseFails("--colorless_only --matching_only Adam"))
    {
        return 6;
    }
    if (!ParseFails("--matching_only Adam"))
    {
        return 7;
    }
    if (!ParseFails("Adam"))
    {
        return 8;
    }
    if (!ParseFails("--colorless_only --matching_window_size 8192 --matching_fft_length 4096 Adam"))
    {
        return 9;
    }
    if (!ParseFails("--colorless_only --matching_min_t60 2 --matching_max_t60 1 Adam"))
    {
        return 10;
    }
    if (!ParseFails("--colorless_only --spectral_fft_size 4096 Adam"))
    {
        return 11;
    }

    auto raw = ValidRawOptions();
    raw.execution.max_time_seconds = -1.0;
    const auto invalid_time = fdn_opt_app::ValidateAndNormalizeCliOptions(raw);
    if (invalid_time || invalid_time.error() != "--max_time_seconds cannot be negative.")
    {
        return 12;
    }

    const std::filesystem::path config_path = "fdn_opt_cli_test.toml";
    {
        std::ofstream config(config_path);
        config << "colorless_only = true\n"
                  "num_channels = 8\n"
                  "gradient_method = \"forward\"\n"
                  "[Adam]\n"
                  "step_size = 0.125\n"
                  "max_iterations = 17\n"
                  "[SPSA]\n"
                  "alpha = 0.4\n";
    }
    const auto from_config = Parse("--config " + config_path.string() + " Adam");
    const bool config_only_failed = ParseFails("--config " + config_path.string());
    std::filesystem::remove(config_path);
    const auto& config_adam = std::get<fdn_optimization::AdamParameters>(from_config.optimizer_params);
    if (from_config.input.fdn_order != 8 ||
        from_config.colorless_execution.gradient_method != fdn_optimization::GradientMethod::ForwardDifferences ||
        std::abs(config_adam.step_size - 0.125f) > 1e-7f || config_adam.max_iterations != 17 ||
        from_config.config_filename != config_path.string() || !config_only_failed)
    {
        return 13;
    }

    const std::filesystem::path invalid_config_path = "fdn_opt_cli_invalid_test.toml";
    {
        std::ofstream config(invalid_config_path);
        config << "colorless_only = true\nunknown_option = 1\n";
    }
    const bool invalid_config_failed = ParseFails("--config " + invalid_config_path.string() + " Adam");
    std::filesystem::remove(invalid_config_path);
    if (!invalid_config_failed)
    {
        return 14;
    }
}
