#!/bin/bash
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    libeigen3-dev \
    libgoogle-glog-dev \
    libboost-all-dev

if [ ! -d "/usr/local/include/ceres" ]; then
    echo "Installing Ceres..."
    git clone https://github.com/ceres-solver/ceres-solver.git
    cd ceres-solver
    git checkout 2.0.0
    mkdir build && cd build
    cmake ..
    make
    sudo make install
    cd ../..
else
    echo "Ceres Solver already installed"
fi

if [ ! -d "/usr/local/include/catch2" ]; then
    git clone https://github.com/catchorg/Catch2.git
    cd Catch2
    cmake -B build -S . -DBUILD_TESTING=OFF
    sudo cmake --build build/ --target install
else
    echo "Catch2 already installed"
fi