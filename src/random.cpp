#include "monte_carlo/random.hpp"
#include <algorithm>

namespace mc {

// MersenneTwisterGenerator Implementation
MersenneTwisterGenerator::MersenneTwisterGenerator()
    : generator_(std::random_device{}())
    , distribution_(0.0, 1.0) {}

MersenneTwisterGenerator::MersenneTwisterGenerator(std::uint64_t seed_value)
    : generator_(seed_value)
    , distribution_(0.0, 1.0) {}

double MersenneTwisterGenerator::generate() {
    return distribution_(generator_);
}

std::vector<double> MersenneTwisterGenerator::generateBatch(size_t n) {
    std::vector<double> result(n);
    std::generate(result.begin(), result.end(),
                 [this]() { return this->generate(); });
    return result;
}

void MersenneTwisterGenerator::seed(std::uint64_t seed_value) {
    generator_.seed(seed_value);
}

// XorShiftGenerator Implementation
XorShiftGenerator::XorShiftGenerator()
    : state_(std::random_device{}()) {
    if (state_ == 0) state_ = 1;
}

XorShiftGenerator::XorShiftGenerator(std::uint64_t seed_value)
    : state_(seed_value) {
    if (state_ == 0) state_ = 1;
}

double XorShiftGenerator::generate() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return static_cast<double>(state_) * INV_MAX;
}

std::vector<double> XorShiftGenerator::generateBatch(size_t n) {
    std::vector<double> result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        result.push_back(generate());
    }
    return result;
}

void XorShiftGenerator::seed(std::uint64_t seed_value) {
    state_ = seed_value;
    if (state_ == 0) state_ = 1;
}

void XorShiftGenerator::discard(unsigned long long count) {
    for (unsigned long long i = 0; i < count; ++i) {
        generate();
    }
}

// Factory Method Implementation
std::unique_ptr<RandomGenerator> RandomGenerator::create(RandomGeneratorType type) {
    switch (type) {
        case RandomGeneratorType::MERSENNE_TWISTER:
            return std::make_unique<MersenneTwisterGenerator>();
        case RandomGeneratorType::XOR_SHIFT:
            return std::make_unique<XorShiftGenerator>();
        default:
            throw std::runtime_error("Unknown random generator type");
    }
}

} // namespace mc