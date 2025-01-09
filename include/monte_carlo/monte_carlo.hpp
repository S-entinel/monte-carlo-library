#ifndef MONTE_CARLO_MONTE_CARLO_HPP
#define MONTE_CARLO_MONTE_CARLO_HPP

#include "monte_carlo/distributions.hpp"
#include "monte_carlo/statistics.hpp"
#include "monte_carlo/simulation_result.hpp"
#include <functional>
#include <future>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace mc {

template<typename T = double>
class MonteCarloUtilities {
public:
    static T standard_error(const std::vector<T>& samples) {
        if (samples.empty()) return T(0);
        
        T mean = std::accumulate(samples.begin(), samples.end(), T(0)) / samples.size();
        T sum_squared_diff = 0;
        
        for (const auto& sample : samples) {
            T diff = sample - mean;
            sum_squared_diff += diff * diff;
        }
        
        T variance = sum_squared_diff / (samples.size() - 1);
        return std::sqrt(variance / samples.size());
    }
    
    static std::vector<T> batch_means(const std::vector<T>& samples, size_t batch_size) {
        if (samples.empty() || batch_size == 0) return {};
        
        std::vector<T> means;
        means.reserve(samples.size() / batch_size);
        
        for (size_t i = 0; i < samples.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, samples.size());
            T batch_sum = std::accumulate(samples.begin() + i, samples.begin() + end, T(0));
            means.push_back(batch_sum / (end - i));
        }
        
        return means;
    }
    
    static std::vector<T> moving_average(const std::vector<T>& samples, size_t window_size) {
        if (samples.empty() || window_size == 0) return {};
        
        std::vector<T> averages;
        averages.reserve(samples.size() - window_size + 1);
        
        T window_sum = std::accumulate(samples.begin(), samples.begin() + window_size, T(0));
        averages.push_back(window_sum / window_size);
        
        for (size_t i = window_size; i < samples.size(); ++i) {
            window_sum = window_sum - samples[i - window_size] + samples[i];
            averages.push_back(window_sum / window_size);
        }
        
        return averages;
    }
};

template<typename T = double>
class MonteCarloEngine {
public:
    using SimulationFunction = std::function<T(const Distribution<T>&)>;
    using ProgressCallback = std::function<void(const SimulationResult<T>&)>;
    
    struct Parameters {
        size_t iterations = 10000;
        size_t num_threads = 1;
        size_t batch_size = 1000;
        bool collect_samples = true;
        bool compute_full_stats = false;
        std::vector<double> quantile_probs = {0.25, 0.5, 0.75};
        std::vector<double> confidence_levels = {0.95};
        double progress_update_interval = 0.1;  // seconds
        ProgressCallback progress_callback = nullptr;
    };
    
    explicit MonteCarloEngine(std::unique_ptr<Distribution<T>> distribution)
        : distribution_(std::move(distribution)) {
        if (!distribution_) {
            throw std::invalid_argument("Distribution cannot be null");
        }
    }
    
    SimulationResult<T> run(const SimulationFunction& sim_func,
                           const Parameters& params = Parameters{}) {
        validate_parameters(params);
        
        SimulationResult<T> result;
        result.iterations = params.iterations;
        result.startTime = std::chrono::steady_clock::now();
        
        // Setup thread pool
        size_t hw_threads = static_cast<size_t>(std::thread::hardware_concurrency());
        if (hw_threads == 0) hw_threads = 1;
        
        size_t num_threads = std::max(size_t(1), std::min(params.num_threads, hw_threads));
            
        size_t iterations_per_thread = params.iterations / num_threads;
        size_t remaining = params.iterations % num_threads;
        
        // Progress tracking
        std::atomic<size_t> completed_iterations{0};
        std::atomic<bool> simulation_running{true};
        
        // Launch progress monitoring thread if callback provided
        std::thread progress_thread;
        if (params.progress_callback) {
            progress_thread = std::thread([&]() {
                monitor_progress(params, completed_iterations, simulation_running, result);
            });
        }
        
        try {
            // Launch simulation threads
            std::vector<std::future<std::vector<T>>> futures;
            futures.reserve(num_threads);
            
            for (size_t i = 0; i < num_threads; ++i) {
                size_t thread_iters = iterations_per_thread + (i == num_threads - 1 ? remaining : 0);
                
                futures.push_back(std::async(std::launch::async,
                    [this, &sim_func, &completed_iterations, thread_iters, &params]() {
                        return run_thread_simulation(sim_func, thread_iters, params.batch_size, completed_iterations);
                    }));
            }
            
            // Collect and process results
            process_simulation_results(futures, params, result);
            
        } catch (...) {
            simulation_running = false;
            if (progress_thread.joinable()) {
                progress_thread.join();
            }
            throw;
        }
        
        // Cleanup and finalize
        simulation_running = false;
        if (progress_thread.joinable()) {
            progress_thread.join();
        }
        
        result.endTime = std::chrono::steady_clock::now();
        result.completionPercentage = 100.0;
        result.estimatedTimeRemaining = 0.0;
        
        return result;
    }
    
    const Distribution<T>& distribution() const { return *distribution_; }
    
    void setDistribution(std::unique_ptr<Distribution<T>> dist) {
        if (!dist) {
            throw std::invalid_argument("Distribution cannot be null");
        }
        distribution_ = std::move(dist);
    }
    
private:
    std::unique_ptr<Distribution<T>> distribution_;
    
    void validate_parameters(const Parameters& params) const {
        if (params.iterations == 0) {
            throw std::invalid_argument("Number of iterations must be positive");
        }
        if (params.batch_size == 0) {
            throw std::invalid_argument("Batch size must be positive");
        }
        if (params.batch_size > params.iterations) {
            throw std::invalid_argument("Batch size cannot exceed total iterations");
        }
        if (params.progress_update_interval <= 0) {
            throw std::invalid_argument("Progress update interval must be positive");
        }
        for (double confidence_level : params.confidence_levels) {
            if (confidence_level <= 0.0 || confidence_level >= 1.0) {
                throw std::invalid_argument("Confidence levels must be between 0 and 1");
            }
        }
        for (double quantile : params.quantile_probs) {
            if (quantile < 0.0 || quantile > 1.0) {
                throw std::invalid_argument("Quantile probabilities must be between 0 and 1");
            }
        }
    }
    
    std::vector<T> run_thread_simulation(const SimulationFunction& sim_func,
                                       size_t thread_iters,
                                       size_t batch_size,
                                       std::atomic<size_t>& completed_iterations) const {
        std::vector<T> thread_results;
        thread_results.reserve(thread_iters);
        
        size_t batches = thread_iters / batch_size;
        size_t remainder = thread_iters % batch_size;
        
        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t j = 0; j < batch_size; ++j) {
                thread_results.push_back(sim_func(*distribution_));
            }
            completed_iterations += batch_size;
        }
        
        for (size_t j = 0; j < remainder; ++j) {
            thread_results.push_back(sim_func(*distribution_));
        }
        completed_iterations += remainder;
        
        return thread_results;
    }
    
    void process_simulation_results(std::vector<std::future<std::vector<T>>>& futures,
                                  const Parameters& params,
                                  SimulationResult<T>& result) const {
        if (params.collect_samples) {
            result.samples.reserve(params.iterations);
        }
        
        std::vector<T> all_samples;
        if (params.compute_full_stats) {
            all_samples.reserve(params.iterations);
        }
        
        T sum = 0;
        T sum_squared = 0;
        size_t total_processed = 0;
        
        for (auto& future : futures) {
            auto thread_results = future.get();
            total_processed += thread_results.size();
            
            for (const auto& value : thread_results) {
                sum += value;
                sum_squared += value * value;
                
                if (params.collect_samples) {
                    result.samples.push_back(value);
                }
                if (params.compute_full_stats) {
                    all_samples.push_back(value);
                }
            }
        }
        
        // Sanity check
        if (total_processed != params.iterations) {
            throw std::runtime_error("Simulation thread count mismatch");
        }
        
        // Calculate basic statistics
        result.mean = sum / params.iterations;
        result.variance = (sum_squared - sum * sum / params.iterations) / 
                         (params.iterations - 1);
        result.standardError = std::sqrt(result.variance / params.iterations);
        
        // Calculate full statistics if requested
        if (params.compute_full_stats) {
            result.fullStats = StatisticalAnalysis<T>::analyze(
                all_samples, params.quantile_probs, params.confidence_levels);
        }
    }
    
    void monitor_progress(const Parameters& params,
                         std::atomic<size_t>& completed_iterations,
                         std::atomic<bool>& simulation_running,
                         SimulationResult<T>& result) const {
        const auto start_time = std::chrono::steady_clock::now();
        bool final_update_sent = false;
        
        while (simulation_running) {
            auto current = completed_iterations.load();
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<double>(now - start_time).count();
            
            result.completionPercentage = (100.0 * current) / params.iterations;
            
            if (current > 0 && elapsed > 0) {
                double rate = current / elapsed;
                result.estimatedTimeRemaining = (params.iterations - current) / rate;
            }
            
            // Wait for initial progress
            if (elapsed < 0.001) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            if (params.progress_callback) {
                params.progress_callback(result);
            }
            
            // Check if we're done
            if (current >= params.iterations) {
                if (!final_update_sent) {
                    result.completionPercentage = 100.0;
                    result.estimatedTimeRemaining = 0.0;
                    if (params.progress_callback) {
                        params.progress_callback(result);
                    }
                    final_update_sent = true;
                }
                break;
            }
            
            // Adaptive sleep interval
            double remaining_percentage = 100.0 - result.completionPercentage;
            double sleep_interval = std::max(0.001, 
                params.progress_update_interval * remaining_percentage / 100.0);
            std::this_thread::sleep_for(
                std::chrono::duration<double>(sleep_interval));
        }
        
        // Force final update if not sent
        if (!final_update_sent && params.progress_callback) {
            result.completionPercentage = 100.0;
            result.estimatedTimeRemaining = 0.0;
            params.progress_callback(result);
        }
    }
};

} // namespace mc

#endif // MONTE_CARLO_MONTE_CARLO_HPP