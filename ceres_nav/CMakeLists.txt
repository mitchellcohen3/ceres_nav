cmake_minimum_required(VERSION 3.10)
project(ceres_nav)

# set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")
SET(GCC_COVERAGE_LINK_FLAGS "-lstdc++fs")
set(CMAKE_CXX_STANDARD 17)

option(OPTION_BUILD_TESTS "Build tests." ON)
option(OPTION_BUILD_EXAMPLES "Build examples." ON) 

# Find Dependancies
find_package(Catch2 3 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time program_options) # for timing

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION})

option(BUILD_TESTS "Build tests" ON)

## Library source files
list(
    APPEND lib_sources
    src/factors/AbsolutePositionFactor.cpp
    src/factors/IMUPreintegrationFactor.cpp
    src/factors/RelativeLandmarkFactor.cpp
    src/factors/RelativePoseFactor.cpp
    src/factors/IMUPriorFactor.cpp

    src/lie/SE23.cpp
    src/lie/SE3.cpp
    src/lie/SO3.cpp

    src/imu/IMUIncrement.cpp
    src/imu/IMUHelper.cpp

    src/utils/Utils.cpp
    src/utils/CostFunctionUtils.cpp

    src/local_parameterizations/PoseLocalParameterization.cpp
    src/local_parameterizations/ExtendedPoseLocalParameterization.cpp

    src/lib/Covariance.cpp
    src/lib/FactorGraph.cpp
    src/lib/StateCollection.cpp
)

# Declare the static library
add_library(${PROJECT_NAME} STATIC ${lib_sources})

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
    ${CERES_LIBRARIES}
)

if (OPTION_BUILD_EXAMPLES) 
    message(STATUS "Building examples")
    add_subdirectory(examples)
endif()
if (OPTION_BUILD_TESTS)
    message(STATUS "Building tests")
    enable_testing()
    add_subdirectory(tests)
endif()

# Install targets
install(TARGETS ${PROJECT_NAME}
    EXPORT ceres_nav_targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)
