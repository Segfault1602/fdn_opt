#pragma once

#include <armadillo>
#include <audio_utils/fft.h>
#include <sffdn/sffdn.h>

#include "audio_loss.h"
#include "loss.h"
#include "optim_types.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace fdn_optimization
{

class LossRegistry
{
  public:
    static LossRegistry& Instance()
    {
        static LossRegistry instance;
        return instance;
    }

    void RegisterLoss(const std::vector<double>& losses)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        losses_ = losses;
    }

    std::vector<double> GetLosses()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return losses_;
    }

  private:
    std::vector<double> losses_;
    std::mutex mutex_;
};

class FDNModel
{
  public:
    FDNModel(sfFDN::FDNConfig initial_config, uint32_t ir_size, std::span<const OptimizationParamType> param_types,
             GradientMethod gradient_method = GradientMethod::CentralDifferences);

    void SetLossFunctions(const std::vector<std::shared_ptr<AudioLoss>>& loss_functions);

    const std::vector<std::shared_ptr<AudioLoss>>& GetLossFunctions() const
    {
        return loss_functions_;
    }

    uint32_t GetParamCount() const;

    void SetGradientDelta(double delta)
    {
        gradient_delta_ = delta;
    }

    void SetEarlyFir(std::span<const float> early_fir)
    {
        early_fir_.assign(early_fir.begin(), early_fir.end());
    }

    void SetT60Estimates(std::span<const float> t60_estimates);

    arma::mat GetInitialParams() const;

    std::vector<float> GenerateIR(const arma::mat& params);

    double Evaluate(const arma::mat& params);
    double Evaluate(const arma::mat& params, const size_t i, const size_t batch_size);

    double EvaluateWithGradient(const arma::mat& x, arma::mat& g);
    double EvaluateWithGradient(const arma::mat& x, const size_t i, arma::mat& g, const size_t batchSize);

    sfFDN::FDNConfig GetFDNConfig(const arma::mat& params) const;

    std::string PrintFDNConfig(const arma::mat& params) const;

    size_t NumFunctions() const
    {
        return 1;
    }

    void Shuffle()
    {
        // No-op
    }

  private:
    sfFDN::FDNConfig initial_config_;
    uint32_t ir_size_;
    std::vector<float> impulse_buffer_;
    std::vector<float> response_buffer_;

    // I don't think there is a reason for these to be shared_ptrs outiside of my own laziness.
    std::vector<std::shared_ptr<AudioLoss>> loss_functions_;

    std::vector<float> matrix_coeffs_;
    std::vector<OptimizationParamType> param_types_;
    std::vector<uint32_t> delays_;
    std::vector<float> t60_estimates_;

    double gradient_delta_ = 1e-3;

    GradientMethod gradient_method_ = GradientMethod::CentralDifferences;
    std::vector<float> early_fir_;

    void GradientCentralDifferences(const arma::mat& x, arma::mat& g);
    void GradientForwardDifferences(const arma::mat& x, arma::mat& g, double current_loss);
};
} // namespace fdn_optimization