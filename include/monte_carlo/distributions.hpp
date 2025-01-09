#ifndef MONTE_CARLO_DISTRIBUTIONS_HPP
#define MONTE_CARLO_DISTRIBUTIONS_HPP

#include "monte_carlo/random.hpp"
#include <cmath>
#include <vector>
#include <memory>
#include <stdexcept>

namespace mc {

template<typename T = double>
class Distribution {
public:
    virtual ~Distribution() = default;
    virtual T sample() const = 0;
    virtual std::vector<T> sampleBatch(size_t n) const = 0;
    virtual T mean() const = 0;
    virtual T variance() const = 0;
    virtual T standardDeviation() const { return std::sqrt(variance()); }
    virtual T skewness() const = 0;
    virtual T kurtosis() const = 0;
    virtual T pdf(T x) const = 0;
    virtual T cdf(T x) const = 0;
    
    virtual void setRandomGenerator(std::unique_ptr<RandomGenerator> generator) {
        if (!generator) {
            throw std::invalid_argument("Random generator cannot be null");
        }
        random_generator_ = std::move(generator);
    }

protected:
    mutable std::unique_ptr<RandomGenerator> random_generator_;
};

template<typename T = double>
class NormalDistribution : public Distribution<T> {
public:
    NormalDistribution(T mean = 0.0, T stddev = 1.0)
        : mean_(mean), stddev_(stddev) {
        if (stddev <= 0) {
            throw std::invalid_argument("Standard deviation must be positive");
        }
    }
    
    T sample() const override {
        if (!this->random_generator_) {
            throw std::runtime_error("Random generator not set");
        }
        
        // Box-Muller transform
        if (!has_cached_value_) {
            double u1 = this->random_generator_->generate();
            double u2 = this->random_generator_->generate();
            
            double mag = stddev_ * std::sqrt(-2.0 * std::log(u1));
            double z1 = mag * std::cos(2 * M_PI * u2) + mean_;
            double z2 = mag * std::sin(2 * M_PI * u2) + mean_;
            
            cached_value_ = z2;
            has_cached_value_ = true;
            return static_cast<T>(z1);
        } else {
            has_cached_value_ = false;
            return static_cast<T>(cached_value_);
        }
    }
    
    std::vector<T> sampleBatch(size_t n) const override {
        if (!this->random_generator_) {
            throw std::runtime_error("Random generator not set");
        }
        
        std::vector<T> result;
        result.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            result.push_back(sample());
        }
        return result;
    }
    
    T mean() const override { return mean_; }
    T variance() const override { return stddev_ * stddev_; }
    T skewness() const override { return 0.0; }
    T kurtosis() const override { return 0.0; }
    
    T pdf(T x) const override {
        T z = (x - mean_) / stddev_;
        return std::exp(-0.5 * z * z) / (stddev_ * std::sqrt(2 * M_PI));
    }
    
    T cdf(T x) const override {
        T z = (x - mean_) / stddev_;
        return 0.5 * (1 + std::erf(z / std::sqrt(2)));
    }

private:
    T mean_;
    T stddev_;
    mutable double cached_value_ = 0.0;
    mutable bool has_cached_value_ = false;
};

template<typename T = double>
class UniformDistribution : public Distribution<T> {
public:
    UniformDistribution(T min = 0.0, T max = 1.0)
        : min_(min), max_(max) {
        if (min >= max) {
            throw std::invalid_argument("Min must be less than max");
        }
    }
    
    T sample() const override {
        if (!this->random_generator_) {
            throw std::runtime_error("Random generator not set");
        }
        return min_ + (max_ - min_) * this->random_generator_->generate();
    }
    
    std::vector<T> sampleBatch(size_t n) const override {
        if (!this->random_generator_) {
            throw std::runtime_error("Random generator not set");
        }
        
        auto uniform_samples = this->random_generator_->generateBatch(n);
        std::vector<T> result;
        result.reserve(n);
        
        for (const auto& u : uniform_samples) {
            result.push_back(min_ + (max_ - min_) * u);
        }
        return result;
    }
    
    T mean() const override { return (min_ + max_) / 2; }
    T variance() const override { return (max_ - min_) * (max_ - min_) / 12; }
    T skewness() const override { return 0.0; }
    T kurtosis() const override { return -6.0 / 5.0; }
    
    T pdf(T x) const override {
        if (x < min_ || x > max_) return 0.0;
        return 1.0 / (max_ - min_);
    }
    
    T cdf(T x) const override {
        if (x < min_) return 0.0;
        if (x > max_) return 1.0;
        return (x - min_) / (max_ - min_);
    }

private:
    T min_;
    T max_;
};

} // namespace mc

#endif // MONTE_CARLO_DISTRIBUTIONS_HPP