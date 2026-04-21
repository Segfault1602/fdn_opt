#include "model.h"

#include <armadillo>
#include <sffdn/sffdn.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <queue>

namespace
{
constexpr uint32_t kNBands = 10;

namespace
{
static std::queue<std::vector<float>> gVectorPool;
static std::mutex gVectorPoolMutex;
std::vector<float> BorrowVector(size_t size)
{
    std::scoped_lock lock(gVectorPoolMutex);
    if (!gVectorPool.empty())
    {
        auto vec = std::move(gVectorPool.front());
        gVectorPool.pop();

        if (vec.size() != size)
        {
            vec.resize(size, 0.0f);
        }

        return vec;
    }
    else
    {
        return std::vector<float>(size, 0.0f);
    }
}

void ReturnVectorToPool(std::vector<float>&& vec)
{
    std::scoped_lock lock(gVectorPoolMutex);
    gVectorPool.push(std::move(vec));
}
} // namespace

// double Sigmoid(double x)
// {
//     return 1.0 / (1.0 + std::exp(-x));
// }

arma::mat ParamToGain(const arma::mat& params)
{
    arma::mat gains = params;
    gains /= arma::norm(gains, 2);
    return gains;
}

arma::mat ParamToGains(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const uint32_t fdn_order = config.fdn_size;
    assert(params.n_cols >= 2 * fdn_order);

    arma::mat input_gains_arma = ParamToGain(params.cols(0, fdn_order - 1));
    arma::mat output_gains_arma = ParamToGain(params.cols(fdn_order, (2 * fdn_order) - 1));

    std::vector<float> input_gains(fdn_order);
    std::vector<float> output_gains(fdn_order);

    for (uint32_t i = 0; i < fdn_order; ++i)
    {
        input_gains[i] = static_cast<float>(input_gains_arma(i));
        output_gains[i] = static_cast<float>(output_gains_arma(i));
    }

    config.input_block_config.parallel_gains_config.gains = std::move(input_gains);
    config.output_block_config.parallel_gains_config.gains = std::move(output_gains);

    const size_t start_offset = 2 * fdn_order;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(2 * fdn_order, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamToMatrix(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const uint32_t fdn_order = config.fdn_size;
    assert(params.n_cols >= fdn_order * fdn_order);

    arma::mat M = params.cols(0, (fdn_order * fdn_order) - 1);
    M.reshape(fdn_order, fdn_order);

    arma::mat Q, R;
    arma::qr_econ(Q, R, M);
    Q = Q * arma::diagmat(arma::sign(R.diag()));

    // arma::mat test = Q.t() * Q;
    // test.print("Q^T * Q:");

    std::vector<float> matrix_coeffs(fdn_order * fdn_order);
    for (uint32_t r = 0; r < fdn_order; ++r)
    {
        for (uint32_t c = 0; c < fdn_order; ++c)
        {
            matrix_coeffs[r * fdn_order + c] = static_cast<float>(Q(r, c));
        }
    }

    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = fdn_order, .type = sfFDN::ScalarMatrixType::Random, .custom_matrix = std::move(matrix_coeffs)};

    const size_t start_offset = fdn_order * fdn_order;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(start_offset, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamToHouseholderMatrix(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const uint32_t fdn_order = config.fdn_size;
    assert(params.n_cols >= fdn_order);

    arma::vec u = params.cols(0, fdn_order - 1).as_col();
    u /= arma::norm(u, 2);

    arma::mat I = arma::eye<arma::mat>(fdn_order, fdn_order);
    arma::mat Q = I - 2.0 * (u * u.t());

    std::vector<float> matrix_coeffs(fdn_order * fdn_order);
    for (uint32_t r = 0; r < fdn_order; ++r)
    {
        for (uint32_t c = 0; c < fdn_order; ++c)
        {
            matrix_coeffs[r * fdn_order + c] = static_cast<float>(Q(r, c));
        }
    }

    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = fdn_order, .type = sfFDN::ScalarMatrixType::Random, .custom_matrix = std::move(matrix_coeffs)};

    const size_t start_offset = fdn_order * fdn_order;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(start_offset, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamToCirculantMatrix(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const uint32_t fdn_order = config.fdn_size;
    assert(params.n_cols >= fdn_order);

    arma::vec u = params.cols(0, fdn_order - 1).as_col();
    arma::cx_vec R = arma::fft(u);
    R = R / arma::abs(R);
    arma::rowvec v = arma::real(arma::ifft(R)).as_row();
    arma::rowvec v2 = arma::shift(arma::fliplr(v), 1);
    arma::mat C = arma::toeplitz(v2, v);

    std::vector<float> matrix_coeffs(fdn_order * fdn_order, 0.0f);
    for (uint32_t r = 0; r < fdn_order; ++r)
    {
        for (uint32_t c = 0; c < fdn_order; ++c)
        {
            matrix_coeffs[r * fdn_order + c] = static_cast<float>(C(r, c));
        }
    }

    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = fdn_order, .type = sfFDN::ScalarMatrixType::Random, .custom_matrix = std::move(matrix_coeffs)};

    const size_t start_offset = fdn_order;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(start_offset, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamsToAttenuationFilters_10Bands(sfFDN::FDNConfig& config, const arma::mat& params)
{
    assert(params.n_cols >= kNBands);

    arma::mat t60s = params.cols(0, kNBands - 1);

    t60s = arma::abs(t60s);
    t60s = arma::clamp(t60s, 0.1, 20.0);

    sfFDN::TenBandFilterOptions attenuation_config;
    attenuation_config.sample_rate = config.sample_rate;
    attenuation_config.shelf_cutoff = 8000.f;

    for (uint32_t i = 0; i < attenuation_config.t60s.size(); ++i)
    {
        attenuation_config.t60s[i] = static_cast<float>(t60s(0, i));
    }

    uint32_t filter_count = 0;
    for (auto& filter : config.loop_filter_configs)
    {
        if (std::holds_alternative<sfFDN::AttenuationFilterBankOptions>(filter))
        {
            ++filter_count;
            auto& filter_bank_config = std::get<sfFDN::AttenuationFilterBankOptions>(filter);
            filter_bank_config.filter_configs.clear();
            for (auto i = 0u; i < config.fdn_size; ++i)
            {
                sfFDN::TenBandFilterOptions filter_config_copy = attenuation_config;
                filter_config_copy.delay = config.delay_bank_config.delays[i];
                filter_bank_config.filter_configs.push_back(filter_config_copy);
            }
        }
    }

    if (filter_count != 1)
    {
        throw std::runtime_error("Expected exactly one AttenuationFilterBankOptions in loop_filter_configs");
    }

    const size_t start_offset = kNBands;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(kNBands, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamsToAttenuationFilters_3Band(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const size_t kParamCount = 3 * config.fdn_size; // 3 for the bands, per channel
    assert(params.n_cols >= kParamCount);

    arma::mat t60 = params.cols(0, kParamCount - 1);

    sfFDN::ThreeBandFilterOptions attenuation_config;
    attenuation_config.freqs = {800.f, 8000.f};
    attenuation_config.q = 1.f / std::numbers::sqrt2_v<float>;
    attenuation_config.sample_rate = config.sample_rate;

    sfFDN::AttenuationFilterBankOptions* filter_bank_config_ptr = nullptr;
    for (auto& filter : config.loop_filter_configs)
    {
        if (std::holds_alternative<sfFDN::AttenuationFilterBankOptions>(filter))
        {
            auto& filter_bank_config = std::get<sfFDN::AttenuationFilterBankOptions>(filter);
            filter_bank_config_ptr = &filter_bank_config;
            break;
        }
    }

    if (!filter_bank_config_ptr)
    {
        throw std::runtime_error("Expected at least one AttenuationFilterBankOptions in loop_filter_configs");
    }

    filter_bank_config_ptr->filter_configs.clear();

    t60 = arma::abs(t60);
    t60 = arma::clamp(t60, 0.1, 20.0);
    for (auto n = 0u; n < config.fdn_size; ++n)
    {
        sfFDN::ThreeBandFilterOptions filter_config_copy = attenuation_config;
        filter_config_copy.delay = config.delay_bank_config.delays[n];
        filter_config_copy.t60s[0] = static_cast<float>(t60(0, n * 3 + 0));
        filter_config_copy.t60s[1] = static_cast<float>(t60(0, n * 3 + 1));
        filter_config_copy.t60s[2] = static_cast<float>(t60(0, n * 3 + 2));
        filter_bank_config_ptr->filter_configs.push_back(filter_config_copy);
    }

    const size_t start_offset = kParamCount;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(kParamCount, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamsToTonecorrectionFilters(sfFDN::FDNConfig& config, const arma::mat& params)
{
    assert(params.n_cols >= kNBands);

    arma::mat gains = params.cols(0, kNBands - 1);

    sfFDN::GraphicEQOptions* eq_config = nullptr;
    for (auto& filter : config.tone_correction_filters)
    {
        if (std::holds_alternative<sfFDN::GraphicEQOptions>(filter))
        {
            eq_config = &std::get<sfFDN::GraphicEQOptions>(filter);
            break;
        }
    }

    if (!eq_config)
    {
        sfFDN::GraphicEQOptions new_eq_config;
        config.tone_correction_filters.push_back(new_eq_config);
        eq_config = &std::get<sfFDN::GraphicEQOptions>(config.tone_correction_filters.back());
    }

    eq_config->freqs = {31.25f, 62.5f, 125.f, 250.f, 500.f, 1000.f, 2000.f, 4000.f, 8000.f, 16000.f};
    eq_config->sample_rate = config.sample_rate;

    for (uint32_t i = 0; i < kNBands; ++i)
    {
        eq_config->gains_db[i] = static_cast<float>(gains(0, i));
    }

    const size_t start_offset = kNBands;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(kNBands, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamsToOverallGain(sfFDN::FDNConfig& config, const arma::mat& params)
{
    assert(params.n_cols >= 1);

    arma::mat gains = params.cols(0, 0);

    for (auto& g : config.output_block_config.parallel_gains_config.gains)
    {
        g *= static_cast<float>(gains(0, 0));
    }

    const size_t start_offset = 1;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(1, params.n_cols - 1);
    return leftover_params;
}

arma::mat GetInitialParamsFromConfig(const sfFDN::FDNConfig& config,
                                     std::span<const fdn_optimization::OptimizationParamType> param_types)
{
    arma::mat params(0, 0);

    const uint32_t fdn_order = config.fdn_size;

    for (const auto& type : param_types)
    {
        switch (type)
        {
        case fdn_optimization::OptimizationParamType::Gains:
        {
            arma::mat input_gains(1, fdn_order);
            arma::mat output_gains(1, fdn_order);

            for (uint32_t i = 0; i < fdn_order; ++i)
            {
                input_gains(0, i) = static_cast<double>(config.input_block_config.parallel_gains_config.gains[i]);
                output_gains(0, i) = static_cast<double>(config.output_block_config.parallel_gains_config.gains[i]);
            }

            params = arma::join_horiz(params, input_gains);
            params = arma::join_horiz(params, output_gains);
        }
        break;
        case fdn_optimization::OptimizationParamType::Matrix:
        {
            arma::mat M(1, fdn_order * fdn_order, arma::fill::randn);

            if (std::holds_alternative<sfFDN::ScalarFeedbackMatrixOptions>(config.feedback_matrix_config))
            {
                const auto& matrix_config = std::get<sfFDN::ScalarFeedbackMatrixOptions>(config.feedback_matrix_config);
                auto matrix = sfFDN::ScalarFeedbackMatrix{matrix_config};

                for (uint32_t r = 0; r < fdn_order; ++r)
                {
                    for (uint32_t c = 0; c < fdn_order; ++c)
                    {
                        M(0, c * fdn_order + r) = static_cast<double>(matrix.GetCoefficient(r, c));
                    }
                }
            }

            params = arma::join_horiz(params, M);
        }
        break;
        case fdn_optimization::OptimizationParamType::Matrix_Householder:
            [[fallthrough]];
        case fdn_optimization::OptimizationParamType::Matrix_Circulant:
        {
            // Using initial config is not supported for the moment
            arma::mat vec_u(1, fdn_order, arma::fill::randn);
            params = arma::join_horiz(params, vec_u);
        }
        break;
        case fdn_optimization::OptimizationParamType::AttenuationFilters:
        {
            arma::mat t60s(1, kNBands, arma::fill::randn);
            t60s = arma::abs(t60s);
            t60s = arma::clamp(t60s, 0.1, 20.0);

            params = arma::join_horiz(params, t60s);
        }
        break;
        case fdn_optimization::OptimizationParamType::AttenuationFilters_3Band:
        {
            arma::mat p(1, 3 * fdn_order, arma::fill::randn);
            p = arma::abs(p);
            p = arma::clamp(p, 0.1, 20.0);

            params = arma::join_horiz(params, p);
        }
        break;
        case fdn_optimization::OptimizationParamType::TonecorrectionFilters:
        {
            arma::mat tc_gains(1, kNBands, arma::fill::randn);
            tc_gains *= 0.5; // start with small gains
            params = arma::join_horiz(params, tc_gains);
        }
        break;
        case fdn_optimization::OptimizationParamType::OverallGain:
        {
            arma::mat overall_gain(1, 1, arma::fill::ones);
            params = arma::join_horiz(params, overall_gain);
        }
        break;
        default:
            throw std::runtime_error("Unknown ParamType in GetParamCount");
        }
    }
    return params;
}

} // namespace

namespace fdn_optimization
{
FDNModel::FDNModel(sfFDN::FDNConfig initial_config, uint32_t ir_size,
                   std::span<const OptimizationParamType> param_types, GradientMethod gradient_method)
    : initial_config_(initial_config)
    , ir_size_(ir_size)
    , param_types_(param_types.begin(), param_types.end())
    , gradient_method_(gradient_method)
{
    const uint32_t fdn_order = initial_config_.fdn_size;

    early_fir_.clear();

    // arma::arma_rng::set_seed_random();

    bool optimize_filters = false;
    for (const auto& type : param_types_)
    {
        if (type == OptimizationParamType::AttenuationFilters || type == OptimizationParamType::TonecorrectionFilters ||
            type == OptimizationParamType::AttenuationFilters_3Band)
        {
            optimize_filters = true;
            break;
        }
    }

    if (!optimize_filters)
    {
        initial_config_.loop_filter_configs.clear();
        sfFDN::AttenuationFilterBankOptions attenuation_filter_bank_config;
        for (uint32_t i = 0; i < fdn_order; ++i)
        {
            sfFDN::HomogenousFilterOptions filter_config;
            filter_config.t60 = 1.f;
            filter_config.delay = initial_config_.delay_bank_config.delays[i];
            filter_config.sample_rate = initial_config_.sample_rate;
            attenuation_filter_bank_config.filter_configs.push_back(filter_config);
        }
        initial_config_.loop_filter_configs.push_back(attenuation_filter_bank_config);

        initial_config_.tone_correction_filters.clear();
    }

    // Check that we only have one type of matrix parameterization
    size_t matrix_param_count = 0;
    for (const auto& type : param_types_)
    {
        if (type == OptimizationParamType::Matrix || type == OptimizationParamType::Matrix_Householder ||
            type == OptimizationParamType::Matrix_Circulant)
        {
            matrix_param_count++;
        }
    }
    if (matrix_param_count > 1)
    {
        throw std::runtime_error("FDNModel only supports one type of matrix parameterization at a time.");
    }
}

void FDNModel::SetLossFunctions(const std::vector<std::shared_ptr<AudioLoss>>& loss_functions)
{
    loss_functions_ = loss_functions;
}

uint32_t FDNModel::GetParamCount() const
{
    uint32_t count = 0;
    const uint32_t fdn_order = initial_config_.fdn_size;
    for (const auto& type : param_types_)
    {
        switch (type)
        {
        case OptimizationParamType::Gains:
            count += 2 * fdn_order;
            break;
        case OptimizationParamType::Matrix:
            count += fdn_order * fdn_order;
            break;
        case OptimizationParamType::Matrix_Householder:
            [[fallthrough]];
        case OptimizationParamType::Matrix_Circulant:
            count += fdn_order;
            break;
        case OptimizationParamType::AttenuationFilters:
            count += kNBands;
            break;
        case OptimizationParamType::AttenuationFilters_3Band:
            count += 3 * fdn_order; // 3 for the bands
            break;
        case OptimizationParamType::TonecorrectionFilters:
            count += kNBands;
            break;
        case OptimizationParamType::OverallGain:
            count += 1;
            break;
        default:
            throw std::runtime_error("Unknown ParamType in GetParamCount");
        }
    }
    return count;
}

void FDNModel::SetT60Estimates(std::span<const float> t60_estimates)
{
    t60_estimates_.assign(t60_estimates.begin(), t60_estimates.end());
}

arma::mat FDNModel::GetInitialParams() const
{
    arma::mat params = GetInitialParamsFromConfig(initial_config_, param_types_);

    assert(params.n_cols == GetParamCount());

    return params;
}

std::vector<float> FDNModel::GenerateIR(const arma::mat& params)
{
    std::vector<float> impulse_buffer = BorrowVector(ir_size_);

    std::vector<float> response_buffer = BorrowVector(ir_size_);

    std::ranges::fill(impulse_buffer, 0.0f);
    std::ranges::fill(response_buffer, 0.0f);

    if (early_fir_.empty())
    {
        impulse_buffer[0] = 1.0f; // Delta impulse
    }
    else
    {
        std::copy(early_fir_.begin(), early_fir_.end(), impulse_buffer.begin());
    }

    sfFDN::AudioBuffer in_buffer(ir_size_, 1, impulse_buffer);
    sfFDN::AudioBuffer out_buffer(ir_size_, 1, response_buffer);
    auto fdn = sfFDN::CreateFDNFromConfig(GetFDNConfig(params));
    fdn->Process(in_buffer, out_buffer);

    ReturnVectorToPool(std::move(impulse_buffer));

    return response_buffer;
}

double FDNModel::Evaluate(const arma::mat& params)
{
    std::vector<float> ir = GenerateIR(params);

    // // Add noise
    // for (auto& sample : ir)
    // {
    //     sample += 1e-5f * static_cast<float>(arma::randn());
    // }

    double total_loss = 0.0;
    std::vector<double> last_losses;
    for (const auto& loss_function : loss_functions_)
    {
        double loss = loss_function->ComputeLoss(ir);
        assert(!std::isnan(loss));
        assert(!std::isinf(loss));
        last_losses.push_back(loss);
        total_loss += loss;
    }

    ReturnVectorToPool(std::move(ir));

    LossRegistry::Instance().RegisterLoss(last_losses);

    return total_loss;
}

double FDNModel::Evaluate(const arma::mat& params, const size_t i, const size_t batch_size)
{
    assert(i == 0 && batch_size == 1);
    (void)i;
    (void)batch_size;
    return Evaluate(params);
}

double FDNModel::EvaluateWithGradient(const arma::mat& x, arma::mat& g)
{
    double loss = Evaluate(x);
    switch (gradient_method_)
    {
    case GradientMethod::CentralDifferences:
        GradientCentralDifferences(x, g);
        break;
    case GradientMethod::ForwardDifferences:
        GradientForwardDifferences(x, g, loss);
        break;
    default:
        throw std::runtime_error("Unknown GradientMethod in EvaluateWithGradient");
    }

    return loss;
}

double FDNModel::EvaluateWithGradient(const arma::mat& x, const size_t i, arma::mat& g, const size_t batchSize)
{
    assert(i == 0 && batchSize == 1);
    (void)i;
    (void)batchSize;
    return EvaluateWithGradient(x, g);
}

sfFDN::FDNConfig FDNModel::GetFDNConfig(const arma::mat& params) const
{
    arma::mat params_to_process = params;
    sfFDN::FDNConfig config = initial_config_;

    for (const auto& type : param_types_)
    {
        switch (type)
        {
        case OptimizationParamType::Gains:
        {
            params_to_process = ParamToGains(config, params_to_process);
        }
        break;
        case OptimizationParamType::Matrix:
        {
            params_to_process = ParamToMatrix(config, params_to_process);
        }
        break;
        case OptimizationParamType::Matrix_Householder:
        {
            params_to_process = ParamToHouseholderMatrix(config, params_to_process);
        }
        break;
        case OptimizationParamType::Matrix_Circulant:
        {
            params_to_process = ParamToCirculantMatrix(config, params_to_process);
        }
        break;
        case OptimizationParamType::AttenuationFilters:
        {
            params_to_process = ParamsToAttenuationFilters_10Bands(config, params_to_process);
        }
        break;
        case OptimizationParamType::AttenuationFilters_3Band:
        {
            params_to_process = ParamsToAttenuationFilters_3Band(config, params_to_process);
        }
        break;
        case OptimizationParamType::TonecorrectionFilters:
        {
            params_to_process = ParamsToTonecorrectionFilters(config, params_to_process);
        }
        break;
        case OptimizationParamType::OverallGain:
        {
            params_to_process = ParamsToOverallGain(config, params_to_process);
        }
        break;
        default:
            throw std::runtime_error("Unknown OptimizationParamType in Setup");
        }
    }

    return config;
}

std::string FDNModel::PrintFDNConfig(const arma::mat& params) const
{
    sfFDN::FDNConfig config = GetFDNConfig(params);

    arma::fvec input_gains_arma(config.input_block_config.parallel_gains_config.gains.data(), config.fdn_size);
    arma::fvec output_gains_arma(config.output_block_config.parallel_gains_config.gains.data(), config.fdn_size);

    std::stringstream ss;

    ss << "FDN Configuration:----------------------" << std::endl;
    ss << "Input Gains: " << std::endl;
    ss << input_gains_arma.t() << std::endl;
    ss << "Output Gains: " << std::endl;
    ss << output_gains_arma.t() << std::endl;
    ss << "Delays: [";
    for (const auto& delay : config.delay_bank_config.delays)
    {
        ss << delay << " ";
    }
    ss << "]" << std::endl;

    // std::vector<float> matrix_data = std::get<std::vector<float>>(config.feedback_matrix_config);
    // arma::fmat matrix_data_arma(matrix_data.data(), config.fdn_size, config.fdn_size);

    // ss << "Feedback Matrix:" << std::endl;
    // ss << matrix_data_arma << std::endl;
    // ss << "----------------------------------------" << std::endl;
    return ss.str();
}

void FDNModel::GradientCentralDifferences(const arma::mat& x, arma::mat& g)
{
    g.zeros(x.n_rows, x.n_cols);

#pragma omp parallel for
    for (int col = 0; col < static_cast<int>(x.n_cols); ++col)
    {
        arma::mat x_plus = x;
        x_plus(0, col) += gradient_delta_;
        // Creating a whole new model to avoid threading issues
        FDNModel grad_model(*this);
        double plus_value = grad_model.Evaluate(x_plus);

        arma::mat x_minus = x;
        x_minus(0, col) -= gradient_delta_;
        double minus_value = grad_model.Evaluate(x_minus);
        g(0, col) = (plus_value - minus_value) / (2 * gradient_delta_);
    }
}

void FDNModel::GradientForwardDifferences(const arma::mat& x, arma::mat& g, double current_loss)
{
    g.zeros(x.n_rows, x.n_cols);
#pragma omp parallel for
    for (int col = 0; col < static_cast<int>(x.n_cols); ++col)
    {
        arma::mat x_plus = x;
        x_plus(0, col) += gradient_delta_;
        // Creating a whole new model to avoid threading issues
        FDNModel grad_model(*this);
        double plus_value = grad_model.Evaluate(x_plus);
        g(0, col) = (plus_value - current_loss) / gradient_delta_;
    }
}

} // namespace fdn_optimization