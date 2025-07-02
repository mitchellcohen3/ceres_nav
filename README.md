# Robotic Navigation in Ceres
This repo contains a set of tools to help implement batch estimators and sliding-window filters for robotic state estimation using Ceres. It contains a set of common factors (such as odometry, vision, and preintegrated IMU factors) used in state estimation, along with common Lie group state definitions such as poses and extended poses. Additionally, both "left" and "right" state definitions are provided for Lie group states, allowing for the user to pick the error definition for Lie group states that best suits the problem at hand.

## Prerequisites
This repo has been tested on Ubuntu 20.04 and has the following requirements:
  - **Ceres Solver 2.0.0** - can be obtained by cloning [this repo](https://github.com/ceres-solver/ceres-solver/tree/2.0.0) and building from source.
  - **Eigen3 >= 3.3**,
  - **[glog](https://github.com/google/glog)**,
  - **[Catch2](https://github.com/catchorg/Catch2)** - to run the unit tests.

## Build
The library, tests, and examples can be built using:
```
git clone https://github.com/mitchellcohen3/ceres_nav/
cd ceres_nav
mkdir build && cd build
cmake ..
make 
make install
```

## Tests 
To ensure that the build functions correctly, tests can be run after installing `Catch2`:
```
cd ceres_nav/build
ctest
```

## Example Usage
This repo contains a simulated GPS/IMU fusion example and can be executed using the Python script `examples/python/run_gps_imu_batch`. The examples additionally depend on the [navlie](https://github.com/decargroup/navlie) Python library, used to generate simulated IMU and GPS measurements, and additionally evaluate the results of the optimization. To install `navlie`, run
```
git clone https://github.com/decargroup/navlie
cd navlie
pip install -e .
```

## Acknowledgements
This repo is inspired partially by [libRSF](https://github.com/TUC-ProAut/libRSF), another great library for robust estimation using Ceres. `libRSF` only supports autodiff cost functions, while this repo supports traditional Ceres cost functions with analytic differentiation and provides tools to additionally check Jacobians numerically. The use of analytical derivatives becomes important when implementing sliding window filters for specific problems, such as visual-inertial odometry where the evaluation point of the Jacobians must be artificially modified to ensure consistency. 

## Disclaimer
Note: this repo is still very much a work in progress, and more complex examples will be added soon! Contributions and thoughts are always welcome :)