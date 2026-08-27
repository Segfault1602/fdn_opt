#pragma once

#include "cli_options.h"

#include <sffdn/sffdn.h>

#include <quill/Logger.h>

#include <filesystem>
#include <span>
#include <tuple>
#include <vector>

namespace fdn_opt_app
{

sfFDN::FDNConfig CreateInitialFDNConfig(uint32_t fdn_order, bool randomize = false, bool random_delays = false,
                                        uint32_t seed = 0);

fdn_optimization::OptimizationResult OptimizeColorless(quill::Logger* logger,
                                                       const sfFDN::FDNConfig& initial_fdn_config,
                                                       const fdn_optimization::OptimizationAlgoParams& optimizer_params,
                                                       const std::tuple<double, double, double>& loss_weights,
                                                       const ExecutionOptions& execution_options, bool verbose,
                                                       MatrixParameterization matrix_parameterization =
                                                           MatrixParameterization::RandomOrthogonal);

fdn_optimization::OptimizationResult OptimizeSpectrum(quill::Logger* logger, const sfFDN::FDNConfig& initial_fdn_config,
                                                      const fdn_optimization::OptimizationAlgoParams& optimizer_params,
                                                      const std::vector<float>& target_rir,
                                                      const std::vector<float>& early_fir,
                                                      const std::tuple<double, double, double>& loss_weights,
                                                      const MatchingAnalysisOptions& matching_options,
                                                      const ExecutionOptions& execution_options, bool verbose);

void RenderAudio(const sfFDN::FDNConfig& fdn_config, const std::string& input_filename,
                 const std::filesystem::path& output_dir, quill::Logger* logger, std::span<const float> early_fir,
                 fdn_optimization::EarlyFirMode early_fir_mode);

} // namespace fdn_opt_app
