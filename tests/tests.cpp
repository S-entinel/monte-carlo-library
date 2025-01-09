#include <gtest/gtest.h>
#include "monte_carlo/random.hpp"
#include "monte_carlo/distributions.hpp"
#include "monte_carlo/monte_carlo.hpp"
#include "monte_carlo/statistics.hpp"
#include <cmath>
#include <algorithm>
#include <mutex>
#include <thread>

using namespace mc;

// Random Generator Tests
class RandomGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        mt_gen = RandomGenerator::create(RandomGeneratorType::MERSENNE_TWISTER);
        xor_gen = RandomGenerator::create(RandomGeneratorType::XOR_SHIFT);
    }

    std::unique_ptr<RandomGenerator> mt_gen;
    std::unique_ptr<RandomGenerator> xor_gen;
};

TEST_F(RandomGeneratorTest, GenerateInRange) {
    for (int i = 0; i < 1000; ++i) {
        double mt_value = mt_gen->generate();
        double xor_value = xor_gen->generate();
        
        EXPECT_GE(mt_value, 0.0);
        EXPECT_LT(mt_value, 1.0);
        EXPECT_GE(xor_value, 0.0);
        EXPECT_LT(xor_value, 1.0);
    }
}

TEST_F(RandomGeneratorTest, BatchGeneration) {
    const size_t batch_size = 1000;
    auto mt_batch = mt_gen->generateBatch(batch_size);
    auto xor_batch = xor_gen->generateBatch(batch_size);
    
    EXPECT_EQ(mt_batch.size(), batch_size);
    EXPECT_EQ(xor_batch.size(), batch_size);
    
    for (const auto& value : mt_batch) {
        EXPECT_GE(value, 0.0);
        EXPECT_LT(value, 1.0);
    }
    for (const auto& value : xor_batch) {
        EXPECT_GE(value, 0.0);
        EXPECT_LT(value, 1.0);
    }
}

// Distribution Tests
class DistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen = RandomGenerator::create();
    }
    
    std::unique_ptr<RandomGenerator> gen;
    const size_t sample_size = 100000;
    const double tolerance = 0.1;  // For statistical tests
};

TEST_F(DistributionTest, NormalDistribution) {
    const double mean = 1.0;
    const double stddev = 2.0;
    
    auto dist = std::make_unique<NormalDistribution<>>(mean, stddev);
    dist->setRandomGenerator(std::move(gen));
    
    auto samples = dist->sampleBatch(sample_size);
    
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double sample_mean = sum / sample_size;
    
    double sum_sq_diff = 0;
    for (const auto& x : samples) {
        double diff = x - sample_mean;
        sum_sq_diff += diff * diff;
    }
    double sample_variance = sum_sq_diff / (sample_size - 1);
    
    EXPECT_NEAR(sample_mean, mean, 0.1);
    EXPECT_NEAR(sample_variance, stddev * stddev, 0.2);
}

TEST_F(DistributionTest, UniformDistribution) {
    const double min = -1.0;
    const double max = 2.0;
    
    auto dist = std::make_unique<UniformDistribution<>>(min, max);
    dist->setRandomGenerator(std::move(gen));
    
    auto samples = dist->sampleBatch(sample_size);
    
    for (const auto& x : samples) {
        EXPECT_GE(x, min);
        EXPECT_LE(x, max);
    }
    
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double sample_mean = sum / sample_size;
    
    double sum_sq_diff = 0;
    for (const auto& x : samples) {
        double diff = x - sample_mean;
        sum_sq_diff += diff * diff;
    }
    double sample_variance = sum_sq_diff / (sample_size - 1);
    
    double expected_mean = (max + min) / 2;
    double expected_variance = (max - min) * (max - min) / 12;
    
    EXPECT_NEAR(sample_mean, expected_mean, 0.1);
    EXPECT_NEAR(sample_variance, expected_variance, 0.1);
}

// Distribution Performance Tests
class DistributionBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        gen = RandomGenerator::create();
    }
    
    std::unique_ptr<RandomGenerator> gen;
    const size_t benchmark_size = 1000000;
};

TEST_F(DistributionBenchmark, CompareDistributionSamplingSpeed) {
    auto normal_dist = std::make_unique<NormalDistribution<>>();
    auto uniform_dist = std::make_unique<UniformDistribution<>>();
    
    normal_dist->setRandomGenerator(RandomGenerator::create());
    uniform_dist->setRandomGenerator(RandomGenerator::create());
    
    auto start = std::chrono::high_resolution_clock::now();
    auto normal_samples = normal_dist->sampleBatch(benchmark_size);
    auto normal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    auto uniform_samples = uniform_dist->sampleBatch(benchmark_size);
    auto uniform_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    std::cout << "\nDistribution Sampling Performance Comparison (" 
              << benchmark_size << " samples):\n";
    std::cout << "Normal Distribution: " << normal_duration << "ms\n";
    std::cout << "Uniform Distribution: " << uniform_duration << "ms\n";
}

// Monte Carlo Engine Core Tests
class MonteCarloEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto dist = std::make_unique<UniformDistribution<>>(0.0, 1.0);
        dist->setRandomGenerator(RandomGenerator::create());
        engine = std::make_unique<MonteCarloEngine<>>(std::move(dist));
    }
    
    std::unique_ptr<MonteCarloEngine<>> engine;
};

TEST_F(MonteCarloEngineTest, BasicIntegration) {
    // Integrate x^2 from 0 to 1
    auto integrand = [](const Distribution<>& dist) {
        double x = dist.sample();
        return x * x;
    };
    
    MonteCarloEngine<>::Parameters params;
    params.iterations = 100000;
    params.num_threads = 4;
    params.compute_full_stats = true;
    
    auto result = engine->run(integrand, params);
    
    EXPECT_NEAR(result.mean, 1.0/3.0, 0.01);
    EXPECT_LT(result.standardError, 0.01);
}

TEST_F(MonteCarloEngineTest, ProgressTracking) {
    std::vector<double> progress_values;
    std::mutex progress_mutex;
    
    MonteCarloEngine<>::Parameters params;
    params.iterations = 10000;
    params.num_threads = 2;
    params.progress_update_interval = 0.01;
    params.batch_size = 50;
    params.progress_callback = [&](const SimulationResult<>& result) {
        std::lock_guard<std::mutex> lock(progress_mutex);
        progress_values.push_back(result.completionPercentage);
    };
    
    auto result = engine->run([](const Distribution<>& dist) { 
        std::this_thread::sleep_for(std::chrono::microseconds(5));
        return dist.sample(); 
    }, params);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    ASSERT_FALSE(progress_values.empty());
    EXPECT_GE(progress_values.back(), 99.0);
    
    for (size_t i = 1; i < progress_values.size(); ++i) {
        EXPECT_GE(progress_values[i], progress_values[i-1]);
    }
}

// Practical Applications Tests
class PracticalApplicationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        normal_dist = std::make_unique<NormalDistribution<>>(0.0, 1.0);
        normal_dist->setRandomGenerator(RandomGenerator::create());
        
        uniform_dist = std::make_unique<UniformDistribution<>>(0.0, 1.0);
        uniform_dist->setRandomGenerator(RandomGenerator::create());
    }
    
    std::unique_ptr<NormalDistribution<>> normal_dist;
    std::unique_ptr<UniformDistribution<>> uniform_dist;
    const size_t default_iterations = 100000;
};

TEST_F(PracticalApplicationsTest, PiEstimation) {
    auto engine = std::make_unique<MonteCarloEngine<>>(std::move(uniform_dist));
    
    auto circle_test = [](const Distribution<>& dist) {
        double x = dist.sample();
        double y = dist.sample();
        return (x*x + y*y <= 1.0) ? 1.0 : 0.0;
    };
    
    MonteCarloEngine<>::Parameters params;
    params.iterations = 1000000;
    params.num_threads = 4;
    params.compute_full_stats = true;
    
    auto result = engine->run(circle_test, params);
    double pi_estimate = result.mean * 4.0;
    
    EXPECT_NEAR(pi_estimate, M_PI, 0.01);
    std::cout << "Pi Estimation: " << pi_estimate << "\n";
    std::cout << "Error: " << std::abs(pi_estimate - M_PI) << "\n";
}

TEST_F(PracticalApplicationsTest, OptionPricing) {
    const double S0 = 100.0;    // Initial stock price
    const double K = 100.0;     // Strike price
    const double r = 0.05;      // Risk-free rate
    const double sigma = 0.2;   // Volatility
    const double T = 1.0;       // Time to maturity
    
    auto engine = std::make_unique<MonteCarloEngine<>>(std::move(normal_dist));
    
    auto european_call = [S0, K, r, sigma, T](const Distribution<>& dist) {
        double z = dist.sample();
        double ST = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);
        return std::exp(-r * T) * std::max(ST - K, 0.0);
    };
    
    MonteCarloEngine<>::Parameters params;
    params.iterations = default_iterations;
    params.compute_full_stats = true;
    
    auto result = engine->run(european_call, params);
    
    // Black-Scholes analytical solution
    double d1 = (std::log(S0/K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    double N_d1 = 0.5 * (1 + std::erf(d1 / std::sqrt(2)));
    double N_d2 = 0.5 * (1 + std::erf(d2 / std::sqrt(2)));
    double bs_price = S0 * N_d1 - K * std::exp(-r * T) * N_d2;
    
    EXPECT_NEAR(result.mean, bs_price, 0.5);
    std::cout << "Option Price (Monte Carlo): " << result.mean << "\n";
    std::cout << "Option Price (Black-Scholes): " << bs_price << "\n";
}

TEST_F(PracticalApplicationsTest, PortfolioRiskAnalysis) {
    const double initial_value = 1000000.0;  // $1M portfolio
    const double mean_return = 0.08;         // 8% annual return
    const double volatility = 0.15;          // 15% volatility
    const double horizon = 1.0/252.0;        // One trading day
    
    auto engine = std::make_unique<MonteCarloEngine<>>(std::move(normal_dist));
    
    auto portfolio_simulation = [&](const Distribution<>& dist) {
        double z = dist.sample();
        double return_ratio = std::exp((mean_return - 0.5 * volatility * volatility) * horizon 
                                     + volatility * std::sqrt(horizon) * z);
        return initial_value * (return_ratio - 1.0);  // Daily P&L
    };
    
    MonteCarloEngine<>::Parameters params;
    params.iterations = default_iterations;
    params.compute_full_stats = true;
    params.quantile_probs = {0.01, 0.05};  // For 99% and 95% VaR
    
    auto result = engine->run(portfolio_simulation, params);
    
    double var_99 = -result.fullStats.quantiles[0];
    double var_95 = -result.fullStats.quantiles[1];
    
    std::cout << "Daily 99% VaR: $" << var_99 << "\n";
    std::cout << "Daily 95% VaR: $" << var_95 << "\n";
    std::cout << "Expected Daily P&L: $" << result.mean << "\n";
}

TEST_F(PracticalApplicationsTest, ReliabilityAnalysis) {
    struct Component {
        double failure_rate;    // failures per hour
        double repair_time;     // hours
    };
    
    std::vector<Component> system = {
        {0.001, 24.0},  // Component 1: fails every 1000 hours, 24h repair
        {0.002, 12.0},  // Component 2: fails every 500 hours, 12h repair
        {0.005, 8.0}    // Component 3: fails every 200 hours, 8h repair
    };
    
    const double simulation_time = 8760.0;  // One year in hours
    
    auto engine = std::make_unique<MonteCarloEngine<>>(std::move(normal_dist));
    
    auto reliability_sim = [&](const Distribution<>& dist) {
        double system_downtime = 0.0;
        
        for (const auto& component : system) {
            double time = 0.0;
            while (time < simulation_time) {
                // Time to next failure using exponential approximation
                double u = std::abs(dist.sample());
                double time_to_failure = -std::log(u) / component.failure_rate;
                
                time += time_to_failure;
                if (time < simulation_time) {
                    system_downtime += component.repair_time;
                    time += component.repair_time;
                }
            }
        }
        
        return (simulation_time - system_downtime) / simulation_time;  // System availability
    };
    
    MonteCarloEngine<>::Parameters params;
    params.iterations = default_iterations / 10;  // Fewer iterations as this is computationally intensive
    params.compute_full_stats = true;
    
    auto result = engine->run(reliability_sim, params);
    
    std::cout << "System Availability: " << result.mean * 100 << "%\n";
    std::cout << "95% Confidence Interval: ["
              << result.confidenceInterval95Low() * 100 << "%, "
              << result.confidenceInterval95High() * 100 << "%]\n";
    
    // Updated expectation based on realistic system configuration
    EXPECT_GT(result.mean, 0.85);  // Expect > 85% availability
    
    // Additional checks for confidence interval width
    double ci_width = result.confidenceInterval95High() - result.confidenceInterval95Low();
    EXPECT_LT(ci_width, 0.01);  // Confidence interval should be reasonably tight
}


TEST_F(PracticalApplicationsTest, MultivariateIntegration) {
    // Compute the volume of a 4D hypersphere
    const int dimension = 4;
    const double radius = 1.0;
    
    auto engine = std::make_unique<MonteCarloEngine<>>(std::move(uniform_dist));
    
    auto hypersphere_test = [dimension, radius](const Distribution<>& dist) {
        double sum_sq = 0.0;
        for (int i = 0; i < dimension; ++i) {
            double x = 2.0 * dist.sample() - 1.0;  // Map to [-1, 1]
            sum_sq += x * x;
        }
        return sum_sq <= radius * radius ? std::pow(2.0, dimension) : 0.0;
    };
    
    MonteCarloEngine<>::Parameters params;
    params.iterations = default_iterations;
    params.compute_full_stats = true;
    
    auto result = engine->run(hypersphere_test, params);
    
    // Analytical volume of n-dimensional unit hypersphere
    double analytical_volume = std::pow(M_PI, dimension/2.0) / std::tgamma(dimension/2.0 + 1.0);
    
    std::cout << "4D Hypersphere Volume (Monte Carlo): " << result.mean << "\n";
    std::cout << "4D Hypersphere Volume (Analytical): " << analytical_volume << "\n";
    std::cout << "Relative Error: " 
              << std::abs(result.mean - analytical_volume) / analytical_volume * 100 << "%\n";
    
    EXPECT_NEAR(result.mean, analytical_volume, analytical_volume * 0.05);  // Within 5%
}

TEST_F(PracticalApplicationsTest, PathDependentOptionPricing) {
    // Asian Option (arithmetic average) pricing
    const double S0 = 100.0;     // Initial stock price
    const double K = 100.0;      // Strike price
    const double r = 0.05;       // Risk-free rate
    const double sigma = 0.2;    // Volatility
    const double T = 1.0;        // Time to maturity
    const int n_steps = 252;     // Daily observations for a year
    
    auto engine = std::make_unique<MonteCarloEngine<>>(std::move(normal_dist));
    
    auto asian_option = [=](const Distribution<>& dist) {
        double dt = T / n_steps;
        double sum_prices = S0;
        double St = S0;
        
        for (int i = 1; i < n_steps; ++i) {
            double z = dist.sample();
            St *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * z);
            sum_prices += St;
        }
        
        double avg_price = sum_prices / n_steps;
        return std::exp(-r * T) * std::max(avg_price - K, 0.0);
    };
    
    MonteCarloEngine<>::Parameters params;
    params.iterations = default_iterations;
    params.compute_full_stats = true;
    params.num_threads = 4;  // Use multiple threads for this intensive calculation
    
    auto result = engine->run(asian_option, params);
    
    std::cout << "Asian Option Price: " << result.mean << "\n";
    std::cout << "Standard Error: " << result.standardError << "\n";
    std::cout << "95% Confidence Interval: ["
              << result.confidenceInterval95Low() << ", "
              << result.confidenceInterval95High() << "]\n";
}

TEST_F(PracticalApplicationsTest, NetworkReliabilityAnalysis) {
    // Monte Carlo analysis of network reliability with redundant paths
    struct Edge {
        int from, to;
        double reliability;
    };
    
    // Simple network with redundant paths from node 0 to node 4
    std::vector<Edge> edges = {
        {0, 1, 0.95}, {0, 2, 0.90},  // Two paths from start
        {1, 3, 0.95}, {2, 3, 0.95},  // Middle connections
        {3, 4, 0.95}, {1, 4, 0.90}   // Two paths to end
    };
    
    auto engine = std::make_unique<MonteCarloEngine<>>(std::move(uniform_dist));
    
    auto network_sim = [&edges](const Distribution<>& dist) {
        std::vector<bool> edge_status(edges.size());
        
        // Simulate edge failures
        for (size_t i = 0; i < edges.size(); ++i) {
            edge_status[i] = (dist.sample() <= edges[i].reliability);
        }
        
        // Check if path exists from 0 to 4 using DFS
        std::vector<std::vector<int>> adj(5);
        for (size_t i = 0; i < edges.size(); ++i) {
            if (edge_status[i]) {
                adj[edges[i].from].push_back(edges[i].to);
            }
        }
        
        std::vector<bool> visited(5, false);
        std::function<bool(int)> dfs = [&](int node) {
            if (node == 4) return true;
            if (visited[node]) return false;
            
            visited[node] = true;
            for (int next : adj[node]) {
                if (dfs(next)) return true;
            }
            return false;
        };
        
        return dfs(0) ? 1.0 : 0.0;
    };
    
    MonteCarloEngine<>::Parameters params;
    params.iterations = default_iterations;
    params.compute_full_stats = true;
    
    auto result = engine->run(network_sim, params);
    
    std::cout << "Network Reliability: " << result.mean * 100 << "%\n";
    std::cout << "95% Confidence Interval: ["
              << result.confidenceInterval95Low() * 100 << "%, "
              << result.confidenceInterval95High() * 100 << "%]\n";
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}