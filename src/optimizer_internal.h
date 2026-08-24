#pragma once

#include "optim_types.h"
#include "parameter_layout.h"

#include <armadillo>

namespace fdn_optimization::detail
{

inline bool ShouldUseStoredBest(const arma::mat& best_coordinates, double best_objective, double final_objective)
{
    return !best_coordinates.is_empty() && best_objective <= final_objective;
}

struct CMAESBounds
{
    arma::mat lower;
    arma::mat upper;
};

inline CMAESBounds BuildMatchingCMAESBounds(const ParameterLayout& layout, const MatchingParameterConfig& matching)
{
    CMAESBounds bounds{
        .lower = arma::mat(1, layout.total_size, arma::fill::value(-1.0)),
        .upper = arma::mat(1, layout.total_size, arma::fill::value(1.0)),
    };
    const bool scaled = matching.parameterization == MatchingParameterization::ScaledSmooth;
    for (const auto& range : layout.ranges)
    {
        double lower = -1.0;
        double upper = 1.0;
        if (range.type == OptimizationParamType::AttenuationFilters ||
            range.type == OptimizationParamType::AttenuationFilters_3Band)
        {
            lower = scaled ? -20.0 : matching.minimum_t60;
            upper = scaled ? 20.0 : matching.maximum_t60;
        }
        else if (range.type == OptimizationParamType::TonecorrectionFilters && !scaled)
        {
            lower = -matching.tone_gain_scale_db;
            upper = matching.tone_gain_scale_db;
        }
        else if (range.type == OptimizationParamType::OverallGain)
        {
            lower = scaled ? -4.0 : 0.0;
            upper = 4.0;
        }
        bounds.lower.cols(range.offset, range.offset + range.size - 1).fill(lower);
        bounds.upper.cols(range.offset, range.offset + range.size - 1).fill(upper);
    }
    return bounds;
}

class CappedMomentumDeltaBarDeltaUpdate
{
  public:
    CappedMomentumDeltaBarDeltaUpdate(double kappa, double phi, double momentum, double min_gain, double max_step_norm)
        : kappa_(kappa)
        , phi_(phi)
        , momentum_(momentum)
        , min_gain_(min_gain)
        , max_step_norm_(max_step_norm)
    {
    }

    template <typename MatType, typename GradType>
    class Policy
    {
      public:
        using ElemType = typename MatType::elem_type;

        Policy(const CappedMomentumDeltaBarDeltaUpdate& parent, std::size_t rows, std::size_t cols)
            : parent_(parent)
        {
            gains_.ones(rows, cols);
            velocity_.zeros(rows, cols);
        }

        void Update(MatType& iterate, double step_size, const GradType& gradient)
        {
            const MatType mismatch = arma::conv_to<MatType>::from(arma::sign(gradient) != arma::sign(velocity_));
            const MatType match = arma::conv_to<MatType>::from(arma::sign(gradient) == arma::sign(velocity_));
            gains_ += mismatch * ElemType(parent_.kappa_) - (match * (ElemType(1) - ElemType(parent_.phi_))) % gains_;
            gains_.clamp(ElemType(parent_.min_gain_), arma::Datum<ElemType>::inf);

            velocity_ = ElemType(parent_.momentum_) * velocity_ - (ElemType(step_size) * gains_) % gradient;
            if (parent_.max_step_norm_ > 0.0)
            {
                const double norm = arma::norm(velocity_, 2);
                if (std::isfinite(norm) && norm > parent_.max_step_norm_)
                    velocity_ *= ElemType(parent_.max_step_norm_ / norm);
            }
            iterate += velocity_;
        }

      private:
        const CappedMomentumDeltaBarDeltaUpdate& parent_;
        MatType gains_;
        MatType velocity_;
    };

  private:
    double kappa_;
    double phi_;
    double momentum_;
    double min_gain_;
    double max_step_norm_;
};

} // namespace fdn_optimization::detail
