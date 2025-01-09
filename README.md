# Monte Carlo Simulation Library

A modern C++ library for running Monte Carlo simulations, featuring multi-threaded execution, multiple probability distributions, and comprehensive statistical analysis capabilities.

## Features

- **Modern C++ Design**
  - Written in C++17 with RAII principles and smart pointers
  - Template metaprogramming for type flexibility
  - Thread-safe implementations

- **Random Number Generation**
  - Multiple generator options (Mersenne Twister, XorShift)
  - Extensible design for custom generators
  - Batch generation support

- **Probability Distributions**
  - Normal (Gaussian) distribution
  - Uniform distribution
  - Extensible framework for additional distributions

- **Monte Carlo Engine**
  - Multi-threaded simulation execution
  - Progress tracking and callbacks
  - Comprehensive statistical analysis
  - Configurable batch processing

## Build Requirements

- C++17 compatible compiler
- CMake 3.15 or higher
- Google Test (automatically fetched by CMake)

## Building the Project

```bash
# Create build directory
mkdir build && cd build

# Generate build files
cmake ..

# Build the project
make

# Run tests
./tests/monte_carlo_tests
```


## Testing

The project includes a comprehensive test suite demonstrating various applications:
- Option pricing
- Portfolio risk analysis
- Network reliability simulation
- Pi estimation
- Integration problems

## Technical Highlights

- **Thread Safety**: Robust multi-threaded implementation with proper synchronization
- **Modern C++ Features**: Smart pointers, RAII, templates, and type safety
- **Error Handling**: Comprehensive exception handling and input validation
- **Build System**: Modern CMake configuration with proper dependency management
- **Testing**: Extensive unit tests using Google Test framework

## Project Structure

```
include/monte_carlo/    # Header files
  ├── distributions.hpp   # Probability distributions
  ├── monte_carlo.hpp    # Main simulation engine
  ├── random.hpp         # Random number generators
  ├── statistics.hpp     # Statistical analysis
  └── simulation_result.hpp
src/                   # Implementation files
tests/                 # Test suite
```
