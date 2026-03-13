#pragma once

#include "model.h"

#include <armadillo>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace fdn_optimization
{

class RandomSearcher
{
  public:
    RandomSearcher();

    template <typename Callback>
    void StartSearch(FDNModel& model, std::stop_token stop_token, double time_limit_seconds, Callback* cb = nullptr)
    {
        const size_t n_threads = 1; // std::thread::hardware_concurrency();
        threads_.clear();
        stop_token_ = stop_token;
        best_params_ = model.GetInitialParams();

        std::atomic<bool> stop_flag{false};

        for (size_t t = 0; t < n_threads; ++t)
        {
            threads_.emplace_back([this, &model, &stop_flag]() {
                FDNModel thread_model = model;

                auto local_best_params = thread_model.GetInitialParams();
                double local_best_objective = std::numeric_limits<double>::infinity();

                arma::mat initial_params = thread_model.GetInitialParams();
                arma::mat params =
                    arma::mat(initial_params.n_rows, initial_params.n_cols, arma::fill::randu) * 2.0 - 1.0;
                // arma::mat gradient;
                // double last_objective = std::numeric_limits<double>::infinity();
                int iteration = 0;
                while (!stop_token_.stop_requested() && !stop_flag.load())
                {

                    double objective = thread_model.Evaluate(params);
                    iteration++;
                    evaluation_count_++;

                    if (objective < local_best_objective)
                    {
                        local_best_objective = objective;
                        local_best_params = params;
                    }

                    if (iteration % 10 == 0)
                    {
                        std::scoped_lock lock(mutex_);
                        if (objective < best_objective_)
                        {
                            best_objective_ = objective;
                            best_params_ = params;
                        }
                    }

                    params = arma::mat(initial_params.n_rows, initial_params.n_cols, arma::fill::randu) * 2.0 - 1.0;
                }

                {
                    std::scoped_lock lock(mutex_);
                    if (local_best_objective < best_objective_)
                    {
                        best_objective_ = local_best_objective;
                        best_params_ = local_best_params;
                    }
                }
            });
        }

        auto start_time = std::chrono::steady_clock::now();
        while (!stop_token_.stop_requested())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            double best_objective;
            arma::mat best_params;
            {
                std::scoped_lock lock(mutex_);
                cb->evaluation_count_ = GetEvaluationCount();
                best_objective = best_objective_;
                best_params = best_params_;
            }

            model.Evaluate(best_params);
            cb->SaveLossHistory(model, best_objective);

            auto elapsed = std::chrono::steady_clock::now() - start_time;
            const std::chrono::duration<double, std::ratio<1>> time_limit(time_limit_seconds);
            if (time_limit_seconds > 0 && elapsed > time_limit)
            {
                stop_flag.store(true);
                for (auto& thread : threads_)
                {
                    if (thread.joinable())
                    {
                        thread.join();
                    }
                }
                break;
            }
        }

        return;
    }

    double GetBestObjective()
    {
        std::scoped_lock lock(mutex_);
        return best_objective_;
    }

    arma::mat GetBestParams()
    {
        std::scoped_lock lock(mutex_);
        return best_params_;
    }

    uint32_t GetEvaluationCount() const
    {
        return evaluation_count_.load();
    }

  private:
    std::mutex mutex_;
    std::stop_token stop_token_;
    double best_objective_;
    arma::mat best_params_;
    std::vector<std::jthread> threads_;
    std::atomic<uint32_t> evaluation_count_{0};
};
} // namespace fdn_optimization