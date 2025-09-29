#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>

#include "lib/ParameterBlock.h"

class ParameterBlockPerformanceFixture {
public:
  ParameterBlockPerformanceFixture() {
    // Create parameter blocks of different sizes
    pose_6d_ = std::make_shared<ParameterBlock<6>>(Eigen::Matrix<double, 6, 1>::Random(), "pose_6d");
    pose_7d_ = std::make_shared<ParameterBlock<7>>(Eigen::Matrix<double, 7, 1>::Random(), "pose_7d");
    point_3d_ = std::make_shared<ParameterBlock<3>>(Eigen::Vector3d::Random(), "point_3d");
    large_12d_ = std::make_shared<ParameterBlock<12>>(Eigen::Matrix<double, 12, 1>::Random(), "large_12d");
    
    // Fill vector with many parameter blocks for bulk operations
    parameter_blocks_.clear();
    parameter_blocks_.reserve(num_blocks_);
    for (int i = 0; i < num_blocks_; ++i) {
      parameter_blocks_.push_back(
        std::make_shared<ParameterBlock<6>>(Eigen::Matrix<double, 6, 1>::Random(), "block_" + std::to_string(i))
      );
    }
  }
  
  template<typename Func>
  double measureTime(Func&& func, int iterations = 100000) {
    // Warm up
    for (int i = 0; i < 1000; ++i) {
      func();
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return static_cast<double>(duration.count()) / iterations;
  }

  std::shared_ptr<ParameterBlock<6>> pose_6d_;
  std::shared_ptr<ParameterBlock<7>> pose_7d_;
  std::shared_ptr<ParameterBlock<3>> point_3d_;
  std::shared_ptr<ParameterBlock<12>> large_12d_;
  std::vector<std::shared_ptr<ParameterBlock<6>>> parameter_blocks_;
  
  static constexpr int num_blocks_ = 1000;
};

TEST_CASE("Parameter Block Access Performance", "[performance]") {
  ParameterBlockPerformanceFixture fixture;
  
  std::cout << "\n=== Parameter Block Access Performance Test ===\n";
  std::cout << std::fixed << std::setprecision(2);
  
  SECTION("3D Point Performance") {
    auto block = fixture.point_3d_;
    std::cout << "\n--- 3D Point (size: " << block->dimension() << ") ---\n";
    
    // 1. Virtual copying access (copy + virtual call)
    double time_virtual_copy = fixture.measureTime([&]() {
      volatile Eigen::VectorXd estimate = block->getEstimate();
      (void)estimate;
    });
    
    // 2. Map access via base class (no copy, virtual call to get pointer)
    double time_map = fixture.measureTime([&]() {
      volatile auto estimate_map = block->getEstimateMap();
      (void)estimate_map;
    });
    
    // 3. Raw pointer access (fastest possible)
    double time_raw = fixture.measureTime([&]() {
      volatile double* ptr = block->estimatePointer();
      (void)ptr;
    });
    
    std::cout << "  Raw pointer access:   " << std::setw(8) << time_raw << " ns\n";
    std::cout << "  Map access:           " << std::setw(8) << time_map << " ns\n";
    std::cout << "  Virtual copy access:  " << std::setw(8) << time_virtual_copy << " ns\n";
    std::cout << "  Copy vs Raw:          " << std::setw(8) << (time_virtual_copy / time_raw) << "x slower\n";
    std::cout << "  Map vs Raw:           " << std::setw(8) << (time_map / time_raw) << "x slower\n";
    
    // Verify the operations work correctly
    REQUIRE(block->dimension() == 3);
    REQUIRE(block->getEstimate().size() == 3);
    REQUIRE(block->getEstimateMap().size() == 3);
  }
  
  SECTION("6D Pose Performance") {
    auto block = fixture.pose_6d_;
    std::cout << "\n--- 6D Pose (size: " << block->dimension() << ") ---\n";
    
    double time_virtual_copy = fixture.measureTime([&]() {
      volatile Eigen::VectorXd estimate = block->getEstimate();
      (void)estimate;
    });
    
    double time_map = fixture.measureTime([&]() {
      volatile auto estimate_map = block->getEstimateMap();
      (void)estimate_map;
    });
    
    double time_raw = fixture.measureTime([&]() {
      volatile double* ptr = block->estimatePointer();
      (void)ptr;
    });
    
    std::cout << "  Raw pointer access:   " << std::setw(8) << time_raw << " ns\n";
    std::cout << "  Map access:           " << std::setw(8) << time_map << " ns\n";
    std::cout << "  Virtual copy access:  " << std::setw(8) << time_virtual_copy << " ns\n";
    std::cout << "  Copy vs Raw:          " << std::setw(8) << (time_virtual_copy / time_raw) << "x slower\n";
    
    REQUIRE(block->dimension() == 6);
  }
  
  SECTION("12D Large State Performance") {
    auto block = fixture.large_12d_;
    std::cout << "\n--- 12D State (size: " << block->dimension() << ") ---\n";
    
    double time_virtual_copy = fixture.measureTime([&]() {
      volatile Eigen::VectorXd estimate = block->getEstimate();
      (void)estimate;
    });
    
    double time_map = fixture.measureTime([&]() {
      volatile auto estimate_map = block->getEstimateMap();
      (void)estimate_map;
    });
    
    double time_raw = fixture.measureTime([&]() {
      volatile double* ptr = block->estimatePointer();
      (void)ptr;
    });
    
    std::cout << "  Raw pointer access:   " << std::setw(8) << time_raw << " ns\n";
    std::cout << "  Map access:           " << std::setw(8) << time_map << " ns\n";
    std::cout << "  Virtual copy access:  " << std::setw(8) << time_virtual_copy << " ns\n";
    std::cout << "  Copy vs Raw:          " << std::setw(8) << (time_virtual_copy / time_raw) << "x slower\n";
    
    REQUIRE(block->dimension() == 12);
  }
}

TEST_CASE("Bulk Operations Performance", "[performance]") {
  ParameterBlockPerformanceFixture fixture;
  
  std::cout << "\n=== Bulk Operations Performance (" << fixture.num_blocks_ << " blocks) ===\n";
  std::cout << std::fixed << std::setprecision(2);
  
  double total_norm = 0.0;
  
  // 1. Using copying interface
  double time_bulk_copy = fixture.measureTime([&]() {
    total_norm = 0.0;
    for (const auto& block : fixture.parameter_blocks_) {
      Eigen::VectorXd estimate = block->getEstimate(); // Copy
      total_norm += estimate.norm();
    }
  }, 1000);
  
  // 2. Using map interface
  double time_bulk_map = fixture.measureTime([&]() {
    total_norm = 0.0;
    for (const auto& block : fixture.parameter_blocks_) {
      auto estimate_map = block->getEstimateMap(); // No copy
      total_norm += estimate_map.norm();
    }
  }, 1000);
  
  std::cout << "  Map access:           " << std::setw(8) << time_bulk_map << " ns per block\n";
  std::cout << "  Copy access:          " << std::setw(8) << time_bulk_copy << " ns per block\n";
  std::cout << "  Copy vs Map:          " << std::setw(8) << (time_bulk_copy / time_bulk_map) << "x slower\n";
  
  // Verify results are consistent
  double norm_copy = 0.0, norm_map = 0.0;
  for (const auto& block : fixture.parameter_blocks_) {
    norm_copy += block->getEstimate().norm();
    norm_map += block->getEstimateMap().norm();
  }
  
  REQUIRE(std::abs(norm_copy - norm_map) < 1e-10);
}

TEST_CASE("Memory Bandwidth Test", "[performance]") {
  std::cout << "\n=== Memory Bandwidth Test ===\n";
  std::cout << std::fixed << std::setprecision(2);
  
  const int large_size = 10000; // Smaller for faster tests
  std::vector<std::shared_ptr<ParameterBlock<6>>> large_blocks;
  large_blocks.reserve(large_size);
  
  for (int i = 0; i < large_size; ++i) {
    large_blocks.push_back(
      std::make_shared<ParameterBlock<6>>(Eigen::Matrix<double, 6, 1>::Random(), "block_" + std::to_string(i))
    );
  }
  
  // Calculate how much data we're moving
  const double data_per_copy_mb = (6 * sizeof(double) * large_size) / (1024.0 * 1024.0);
  
  // Test copying bandwidth
  auto start = std::chrono::high_resolution_clock::now();
  double sum = 0.0;
  for (const auto& block : large_blocks) {
    Eigen::VectorXd copy = block->getEstimate(); // Copy
    sum += copy.sum(); // Use the data to prevent optimization
  }
  auto end = std::chrono::high_resolution_clock::now();
  
  auto copy_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  double copy_bandwidth_mb_s = data_per_copy_mb / (copy_time_ms / 1000.0);
  
  // Test no-copy bandwidth
  start = std::chrono::high_resolution_clock::now();
  double sum2 = 0.0;
  for (const auto& block : large_blocks) {
    auto direct = block->getEstimateMap(); // No copy
    sum2 += direct.sum();
  }
  end = std::chrono::high_resolution_clock::now();
  
  auto direct_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  
  std::cout << "  Data size:            " << std::setw(8) << data_per_copy_mb << " MB\n";
  std::cout << "  Copy time:            " << std::setw(8) << copy_time_ms << " ms\n";
  std::cout << "  Direct time:          " << std::setw(8) << direct_time_ms << " ms\n";
  std::cout << "  Copy bandwidth:       " << std::setw(8) << copy_bandwidth_mb_s << " MB/s\n";
  std::cout << "  Slowdown factor:      " << std::setw(8) << (copy_time_ms / direct_time_ms) << "x\n";
  
  // Verify correctness
  REQUIRE(std::abs(sum - sum2) < 1e-10);
  REQUIRE(copy_time_ms > 0);
  REQUIRE(direct_time_ms > 0);
}

TEST_CASE("Realistic Usage Patterns", "[performance]") {
  ParameterBlockPerformanceFixture fixture;
  std::cout << "\n=== Realistic Usage Patterns ===\n";
  
  // Pattern: Frequent access in optimization loop (quadratic cost function)
  const int iterations = 1000;
  
  double time_optimization_loop_copy = fixture.measureTime([&]() {
    double cost = 0.0;
    for (const auto& block : fixture.parameter_blocks_) {
      Eigen::VectorXd state = block->getEstimate(); // Copy
      cost += (state.transpose() * state)(0,0); // Quadratic cost
    }
  }, iterations);
  
  double time_optimization_loop_map = fixture.measureTime([&]() {
    double cost = 0.0;
    for (const auto& block : fixture.parameter_blocks_) {
      auto state = block->getEstimateMap(); // No copy
      cost += state.squaredNorm(); // Quadratic cost
    }
  }, iterations);
  
  std::cout << "  Optimization loop (copy): " << time_optimization_loop_copy << " ns\n";
  std::cout << "  Optimization loop (map):  " << time_optimization_loop_map << " ns\n";
  std::cout << "  Slowdown factor:          " << (time_optimization_loop_copy / time_optimization_loop_map) << "x\n";
  
  // Verify both methods give same results
  double cost1 = 0.0, cost2 = 0.0;
  for (const auto& block : fixture.parameter_blocks_) {
    Eigen::VectorXd state1 = block->getEstimate();
    auto state2 = block->getEstimateMap();
    cost1 += (state1.transpose() * state1)(0,0);
    cost2 += state2.squaredNorm();
  }
  REQUIRE(std::abs(cost1 - cost2) < 1e-10);
}

// Optional: Catch2 benchmark integration (if you have Catch2 v3+)
#ifdef CATCH_CONFIG_ENABLE_BENCHMARKING
TEST_CASE("Benchmark Parameter Access", "[!benchmark]") {
  auto block = std::make_shared<ParameterBlock<6>>(Eigen::Matrix<double, 6, 1>::Random(), "test");
  
  BENCHMARK("Virtual copy access") {
    return block->getEstimate();
  };
  
  BENCHMARK("Map access") {
    return block