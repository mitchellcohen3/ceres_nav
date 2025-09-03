# Robotic Navigation in Ceres
![CI](https://github.com/mitchellcohen3/ceres_nav/workflows/CI/badge.svg)

This repo contains a set of tools to help implement batch estimators and sliding-window filters for robotic state estimation using Ceres. It contains a set of common factors (such as odometry, vision, and preintegrated IMU factors) used in state estimation, along with common Lie group state definitions such as poses and extended poses. Additionally, both "left" and "right" state definitions are provided for Lie group states, allowing for the user to pick the error definition for Lie group states that best suits the problem at hand.

## Prerequisites
This repo has been tested on Ubuntu 20.04 and has the following requirements:
  - **Ceres Solver 2.0.0** - can be built from source using
  ```bash
  $ git clone git@github.com:ceres-solver/ceres-solver.git && cd ceres-solver
  $ git checkout 2.0.0
  $ mkdir build && cd build
  $ cmake ..
  $ make 
  $ make install
  ``` 
  - **Eigen3 >= 3.3**,
  ```bash
  $ sudo apt install libeigen3-dev
  ```
  - **[glog](https://github.com/google/glog)**,
  ```bash
  $ sudo apt install libgoogle-glog-dev
  ```
  - **[Catch2](https://github.com/catchorg/Catch2)** - to run the unit tests.
    Catch2 can be installed from source using
  ```bash
  $ git clone https://github.com/catchorg/Catch2.git
  $ cd Catch2
  $ cmake -B build -S . -DBUILD_TESTING=OFF
  $ sudo cmake --build build/ --target install
  ```
  - **Boost** - for filesystem and program options. Can be installed using
  ```bash
  $ sudo apt install libboost-all-dev
  ```

The dependencies can be also installed using the provided `scripts/install_dependencies.sh` script.

## Build
After installing the dependencies, the `ceres_nav` library, tests, and examples can be built using
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
An simulated example of running batch estimators and sliding window filters for GPS/IMU fusion is provided in `gps_imu_example.cpp`. In this example, the state to be estimated is the IMU state, consisting of
- Orientation, $\mathbf{C}_{ab} \in SO(3)$,
- Inertial velocity, $\mathbf{v}_a \in \mathbb{R}^3$
- Inertial position, $\mathbf{r}_a \in \mathbb{R}^3$,
- IMU biases $\mathbf{b}_b \in \mathbb{R}^6$.

This example showcases how this library can be utilized to test various state representations for estimation problems. For example, the orientation, velocity, and position can be parameterized as an element of $SE_2(3)$ as


<p>
$$
\mathbf{T}_{ab} = 
\left[\begin{array}{ccc}
\mathbf{C}_{ab} & \mathbf{v}_a^{zw} & \mathbf{r}_a^{zw} \\
\mathbf{0} & 1 & 0 \\
\mathbf{0} & 0 & 1
\end{array}\right] \in SE_2(3),
$$
</p>

where either a "left" or "right" perturbation scheme can be used. Additionally, a user can select a more traditional state representation, $SO(3) \times \mathbb{R}^3 \times \mathbb{R}^3$, where a multiplicative error is used for the orientation and additive errors are used for the velocity and position. 

To see how to run estimators using these various configurations, see the script `examples/python/run_gps_imu_fusion.py`, which can be run as
```bash
$ python3 examples/python/run_gps_imu_fusion.py \ 
  --lie_direction "left" \
  --state_representation "decoupled"
```
which can be run for both left and right Lie directions or and both $SE_2(3)$ and decoupled state representations.

This additionally depends on the [navlie](https://github.com/decargroup/navlie) Python library, used to generate simulated IMU and GPS measurements, and additionally evaluate the results of the optimization. To install `navlie`, run
```
$ git clone https://github.com/decargroup/navlie
$ cd navlie
$ pip install -e .
```

In this example, the state to be estimated is the IMU state, consisting of orientation $\mathbf{C}_{ab} \in SO(3)$, velocity $\mathbf{v}_a$, position $\mathbf{r}_a$, gyroscope bias, and accelerometer bias. This example showcases how the library allows for easily changing the state representation. For example, the user can either represent the state as an element of 

## Acknowledgements
This repo is inspired partially by [libRSF](https://github.com/TUC-ProAut/libRSF), another great library for robust estimation using Ceres. `libRSF` only supports autodiff cost functions, while this repo supports traditional Ceres cost functions with analytic differentiation and provides tools to additionally check Jacobians numerically. The use of analytical derivatives becomes important when implementing sliding window filters for specific problems, such as visual-inertial odometry where the evaluation point of the Jacobians must be artificially modified to ensure consistency. 

## Disclaimer
Note: this repo is still very much a work in progress, and more complex examples will be added soon! Contributions and thoughts are always welcome :)