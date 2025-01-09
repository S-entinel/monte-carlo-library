#ifndef MONTE_CARLO_SIMULATION_RESULT_HPP
#define MONTE_CARLO_SIMULATION_RESULT_HPP

#include "monte_carlo/statistics.hpp"
#include <vector>
#include <chrono>

namespace mc {

template<typename T = double>
struct SimulationResult {
    T mean = 0;
    T variance = 0;
    T standardError = 0;
    size_t iterations = 0;
    std::vector<T> samples;
    typename StatisticalAnalysis<T>::Statistics fullStats;
    double completionPercentage = 0.0;
    double estimatedTimeRemaining = 0.0;  // in seconds
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;
    
    T confidenceInterval95Low() const { return mean - 1.96 * standardError; }
    T confidenceInterval95High() const { return mean + 1.96 * standardError; }
    
    double totalDuration() const {
        return std::chrono::duration<double>(endTime - startTime).count();
    }
};

} // namespace mc

#endif // MONTE_CARLO_SIMULATION_RESULT_HPP