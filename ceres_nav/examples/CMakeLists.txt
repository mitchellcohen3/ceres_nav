cmake_minimum_required(VERSION 3.10)
project(ceres_swf_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(sources
    gps_imu_batch.cpp
    include/GPSIMUExampleUtils.h
)

# Create executable
add_executable(gps_imu_batch ${sources})
target_include_directories(gps_imu_batch PRIVATE include)

# Link the example to the main library and dependancies
target_link_libraries(gps_imu_batch ceres_nav ${CERES_LIBRARIES} ${Boost_LIBRARIES})
