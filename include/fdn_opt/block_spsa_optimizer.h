#pragma once

#include "model.h"
#include "optimizer.h"
#include "parameter_layout.h"

#include <armadillo>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stop_token>
#include <string>
#include <vector>

namespace fdn_optimization
{

// Describes the probe and update diagnostics for one block within an accepted update.
struct BlockSPSAProbe
{
    std::size_t block = 0;
    std::size_t block_size = 0;
    std::size_t visit = 0;
    double learning_rate = 0.0;
    double perturbation = 0.0;
    // Mean loss over the replications probed at x + c * delta and x - c * delta.
    double probe_plus = 0.0;
    double probe_minus = 0.0;
    // Mean and mean-absolute paired difference f(x + c * delta) - f(x - c * delta).
    double paired_difference = 0.0;
    double absolute_paired_difference = 0.0;
    double gradient_norm = 0.0;
    double step_norm = 0.0;
};

// Describes one accepted BlockSPSA update.
struct BlockSPSAStep
{
    std::size_t step = 0;
    ObjectiveEvaluation evaluation;
    double best_loss = 0.0;
    double learning_rate = 0.0;
    double gradient_norm = 0.0;
    std::uint64_t objective_evaluations = 0;
    std::chrono::duration<double> elapsed_time;
    std::optional<std::size_t> active_block;
    std::optional<std::size_t> block_visit;
    double perturbation = 0.0;
    std::size_t directions_averaged = 1;
    // False when the accepted point was not evaluated because of the accepted-evaluation interval.
    bool evaluated = true;
    // True when this accepted update produced a new best evaluated point.
    bool improved_best = false;
    double step_norm = 0.0;
    std::vector<BlockSPSAProbe> block_probes;
};

// Contains the final accepted state of a BlockSPSA run.
struct BlockSPSAResult
{
    arma::mat parameters;
    ObjectiveEvaluation evaluation;
    std::size_t accepted_updates = 0;
    std::size_t block_count = 0;
    double stability_constant = 0.0;
    std::size_t stall_window = 0;
    std::string termination_reason = "converged";
};

// Project-local block-coordinate simultaneous perturbation optimizer.
class BlockSPSAOptimizer
{
  public:
    using ObjectiveFunction = std::function<ObjectiveEvaluation(const arma::mat&, bool)>;
    using StepCallback = std::function<void(const BlockSPSAStep&)>;

    explicit BlockSPSAOptimizer(BlockSPSAParameters parameters);

    BlockSPSAResult Optimize(ObjectiveFunction objective, arma::mat initial_parameters,
                             ObjectiveEvaluation initial_evaluation, std::span<const ParameterBlock> blocks,
                             std::uint32_t seed, std::uint32_t probe_threads, std::uint64_t max_objective_evaluations,
                             double max_time_seconds, std::stop_token stop_token,
                             StepCallback step_callback = {}) const;

    BlockSPSAResult Optimize(FDNModel& model, arma::mat initial_parameters, ObjectiveEvaluation initial_evaluation,
                             std::span<const ParameterBlock> blocks, std::uint32_t seed, std::uint32_t probe_threads,
                             std::uint64_t max_objective_evaluations, double max_time_seconds,
                             std::stop_token stop_token, StepCallback step_callback = {}) const;

  private:
    BlockSPSAParameters parameters_;
};

} // namespace fdn_optimization
