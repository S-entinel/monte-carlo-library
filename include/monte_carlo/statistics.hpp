#ifndef MONTE_CARLO_STATISTICS_HPP
#define MONTE_CARLO_STATISTICS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace mc {

namespace detail {
    // Implementation of inverse error function
    inline double erfinv(double x) {
        if (x < -1 || x > 1) {
            throw std::domain_error("erfinv requires -1 <= x <= 1");
        }
        
        if (x == 0) return 0;
        
        // Approximation for |x| <= 0.7
        if (std::abs(x) <= 0.7) {
            const double a[4] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
            const double b[4] = {-2.118377725, 1.442710462, -0.329097515, 0.012229801};
            double y = x * x;
            double num = ((a[3] * y + a[2]) * y + a[1]) * y + a[0];
            double den = ((b[3] * y + b[2]) * y + b[1]) * y + b[0];
            return x * num / den;
        }
        
        // Approximation for |x| > 0.7
        double y = std::sqrt(-std::log((1 - std::abs(x)) / 2));
        double z = (((-0.000200214257 * y + 0.000100950558) * y + 0.00134934322) * y - 0.00367342844) * y + 
                   0.00573950773 - (((-0.000200214257 * y + 0.000100950558) * y + 0.00134934322) * y - 
                   0.00367342844) * y - 0.00573950773;
        
        return (x >= 0 ? y + z : -y - z);
    }
}

template<typename T>
class StatisticalAnalysis {
public:
    struct Statistics {
        T mean;
        T median;
        T variance;
        T standardDeviation;
        T skewness;
        T kurtosis;
        T standardError;
        std::vector<T> quantiles;
        std::vector<T> confidenceIntervals;
        size_t sampleSize;
    };
    
    static Statistics analyze(const std::vector<T>& data,
                            const std::vector<double>& quantile_probs = {0.25, 0.5, 0.75},
                            const std::vector<double>& confidence_levels = {0.95}) {
        if (data.empty()) {
            throw std::invalid_argument("Empty data set");
        }
        
        Statistics stats;
        stats.sampleSize = data.size();
        stats.mean = calculate_mean(data);
        stats.median = calculate_median(data);
        stats.variance = calculate_variance(data, stats.mean);
        stats.standardDeviation = std::sqrt(stats.variance);
        stats.skewness = calculate_skewness(data, stats.mean, stats.standardDeviation);
        stats.kurtosis = calculate_kurtosis(data, stats.mean, stats.standardDeviation);
        stats.standardError = stats.standardDeviation / std::sqrt(data.size());
        stats.quantiles = calculate_quantiles(data, quantile_probs);
        stats.confidenceIntervals = calculate_confidence_intervals(
            stats.mean, stats.standardError, confidence_levels);
        
        return stats;
    }

private:
    static T calculate_mean(const std::vector<T>& data) {
        return std::accumulate(data.begin(), data.end(), T(0)) / static_cast<T>(data.size());
    }
    
    static T calculate_median(const std::vector<T>& data) {
        std::vector<T> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());
        
        if (sorted_data.size() % 2 == 0) {
            size_t mid = sorted_data.size() / 2;
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2;
        }
        return sorted_data[sorted_data.size() / 2];
    }
    
    static T calculate_variance(const std::vector<T>& data, T mean) {
        T sum_sq_diff = 0;
        for (const auto& x : data) {
            T diff = x - mean;
            sum_sq_diff += diff * diff;
        }
        return sum_sq_diff / (data.size() - 1);
    }
    
    static T calculate_skewness(const std::vector<T>& data, T mean, T std_dev) {
        T sum_cubed_diff = 0;
        for (const auto& x : data) {
            T diff = (x - mean) / std_dev;
            sum_cubed_diff += diff * diff * diff;
        }
        return sum_cubed_diff / data.size();
    }
    
    static T calculate_kurtosis(const std::vector<T>& data, T mean, T std_dev) {
        T sum_fourth_diff = 0;
        for (const auto& x : data) {
            T diff = (x - mean) / std_dev;
            sum_fourth_diff += diff * diff * diff * diff;
        }
        return sum_fourth_diff / data.size() - 3;
    }
    
    static std::vector<T> calculate_quantiles(const std::vector<T>& data,
                                            const std::vector<double>& probs) {
        std::vector<T> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());
        
        std::vector<T> quantiles;
        quantiles.reserve(probs.size());
        
        for (double p : probs) {
            double idx = p * (sorted_data.size() - 1);
            size_t i = static_cast<size_t>(idx);
            double frac = idx - i;
            
            if (i + 1 >= sorted_data.size()) {
                quantiles.push_back(sorted_data.back());
            } else {
                quantiles.push_back(sorted_data[i] * (1 - frac) + 
                                  sorted_data[i + 1] * frac);
            }
        }
        
        return quantiles;
    }
    
    static std::vector<T> calculate_confidence_intervals(T mean, T std_error,
                                                       const std::vector<double>& levels) {
        std::vector<T> intervals;
        intervals.reserve(levels.size() * 2);
        
        for (double level : levels) {
            double z = std::sqrt(2) * detail::erfinv(level);
            intervals.push_back(mean - z * std_error);
            intervals.push_back(mean + z * std_error);
        }
        
        return intervals;
    }
};

} // namespace mc

#endif // MONTE_CARLO_STATISTICS_HPP