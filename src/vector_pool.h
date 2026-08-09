#pragma once

#include <mutex>
#include <queue>
#include <vector>

namespace fdn_optimization
{
class VectorPool
{
  public:
    static VectorPool& Instance()
    {
        static VectorPool instance;
        return instance;
    }

    [[nodiscard]] std::vector<float> BorrowVector(size_t size)
    {
        std::scoped_lock lock(mutex_);
        if (!pool_.empty())
        {
            auto vec = std::move(pool_.front());
            pool_.pop();

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

    void ReturnVector(std::vector<float>&& vec)
    {
        std::scoped_lock lock(mutex_);
        pool_.push(std::move(vec));
    }

  private:
    std::queue<std::vector<float>> pool_;
    std::mutex mutex_;
};
} // namespace fdn_optimization