#pragma once

#include "cli_options.h"

#include <sffdn/sffdn.h>

#include <quill/Logger.h>

#include <nlohmann/json_fwd.hpp>

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace fdn_opt_app
{

class OptimizationApplication
{
  public:
    OptimizationApplication(quill::Logger* logger, ParsedCliOptions options);

    int Run();

  private:
    void ConfigureLoggingAndReportRun();
    void PrepareOutputDirectory();
    std::expected<sfFDN::FDNConfig, std::string> CreateOrLoadInitialConfig();
    void SaveInitialArtifacts(const sfFDN::FDNConfig& config) const;
    std::expected<fdn_optimization::OptimizationResult, std::string> RunColorlessStage(
        const sfFDN::FDNConfig& initial_config) const;
    bool WriteColorlessOutputs(const fdn_optimization::OptimizationResult& result) const;
    std::expected<std::vector<float>, std::string> LoadTargetRir() const;
    std::expected<std::vector<float>, std::string> LoadEarlyFir(size_t target_size) const;
    std::expected<std::vector<float>, std::string> LoadMonoAudio(const std::string& path, const std::string& label,
                                                                 const std::string& failure_message) const;
    std::expected<fdn_optimization::OptimizationResult, std::string> OptimizeMatching(
        const sfFDN::FDNConfig& initial_config, const std::vector<float>& target_rir,
        const std::vector<float>& early_fir) const;
    bool WriteMatchingOutputs(const fdn_optimization::OptimizationResult& result, const std::vector<float>& target_rir,
                              const std::vector<float>& early_fir) const;
    bool WriteTargetRirName() const;
    void RenderExampleAudio(const sfFDN::FDNConfig& config, const std::vector<float>& early_fir) const;
    void LogStageResult(const char* stage, const fdn_optimization::OptimizationResult& result) const;
    nlohmann::json BuildColorlessMetadata() const;
    nlohmann::json BuildMatchingMetadata(size_t target_size) const;

    quill::Logger* logger_;
    ParsedCliOptions options_;
    std::filesystem::path output_directory_;
    uint32_t fdn_order_;
};

} // namespace fdn_opt_app
