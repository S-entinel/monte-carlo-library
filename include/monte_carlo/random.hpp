#ifndef MONTE_CARLO_RANDOM_HPP
#define MONTE_CARLO_RANDOM_HPP

#include <random>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>

namespace mc {

enum class RandomGeneratorType {
    MERSENNE_TWISTER,
    XOR_SHIFT
};

class RandomGenerator {
public:
    virtual ~RandomGenerator() = default;
    virtual double generate() = 0;
    virtual std::vector<double> generateBatch(size_t n) = 0;
    virtual void seed(std::uint64_t seed_value) = 0;
    virtual std::string name() const = 0;
    virtual void discard(unsigned long long count) = 0;
    
    static std::unique_ptr<RandomGenerator> create(
        RandomGeneratorType type = RandomGeneratorType::MERSENNE_TWISTER);
};

class MersenneTwisterGenerator : public RandomGenerator {
public:
    MersenneTwisterGenerator();
    explicit MersenneTwisterGenerator(std::uint64_t seed_value);
    
    double generate() override;
    std::vector<double> generateBatch(size_t n) override;
    void seed(std::uint64_t seed_value) override;
    std::string name() const override { return "Mersenne Twister"; }
    void discard(unsigned long long count) override { generator_.discard(count); }

private:
    std::mt19937_64 generator_;
    std::uniform_real_distribution<double> distribution_;
};

class XorShiftGenerator : public RandomGenerator {
public:
    XorShiftGenerator();
    explicit XorShiftGenerator(std::uint64_t seed_value);
    
    double generate() override;
    std::vector<double> generateBatch(size_t n) override;
    void seed(std::uint64_t seed_value) override;
    std::string name() const override { return "XorShift"; }
    void discard(unsigned long long count) override;

private:
    std::uint64_t state_;
    static constexpr double INV_MAX = 1.0 / static_cast<double>(UINT64_MAX);
};

} // namespace mc

#endif // MONTE_CARLO_RANDOM_HPP