cmake_minimum_required(VERSION 3.10)
project(ceres_nav)

# set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")
SET(GCC_COVERAGE_LINK_FLAGS "-lstdc++fs")
set(CMAKE_CXX_STANDARD 17)

option(OPTION_BUILD_TESTS "Build tests." ON)
option(OPTION_BUILD_EXAMPLES "Build examples." ON) 

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS: "Build type: " ${CMAKE_BUILD_TYPE})

# Find Dependancies
find_package(Catch2 3 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres 2.0.0 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time program_options) # for timing

# Lie++ library
include(FetchContent)
FetchContent_Declare(
    LiePlusPlus
    GIT_REPOSITORY  https://github.com/aau-cns/Lie-plusplus
    GIT_TAG         main
    GIT_SHALLOW     TRUE
    GIT_PROGRESS    TRUE
)
FetchContent_MakeAvailable(LiePlusPlus)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION})
message(STATUS "Lie++ VER: " ${LiePlusPlus_VERSION})
option(BUILD_TESTS "Build tests" ON)

## Library source files
list(
    APPEND lib_sources
    src/factors/AbsolutePositionFactor.cpp
    src/factors/IMUPreintegrationFactor.cpp
    src/factors/RelativeLandmarkFactor.cpp
    src/factors/RelativePoseFactor.cpp
    src/factors/IMUPriorFactor.cpp
    src/factors/MarginalizationPrior.cpp

    src/lie/SE23.cpp
    src/lie/SE3.cpp
    src/lie/SO3.cpp

    src/imu/IMUIncrement.cpp
    src/imu/IMUHelper.cpp
    src/imu/IMUPreintegrationHelper.cpp

    src/utils/Utils.cpp
    src/utils/CostFunctionUtils.cpp
    src/utils/VectorMath.cpp

    src/local_parameterizations/SO3LocalParameterization.cpp
    src/local_parameterizations/PoseLocalParameterization.cpp
    src/local_parameterizations/ExtendedPoseLocalParameterization.cpp
    src/local_parameterizations/SE23LocalParameterization.cpp
    src/local_parameterizations/DecoupledExtendedPoseLocalParameterization.cpp

    src/lib/Covariance.cpp
    src/lib/FactorGraph.cpp
    src/lib/StateCollection.cpp
    src/lib/Marginalization.cpp
)

# Declare the static library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(${PROJECT_NAME} STATIC ${lib_sources})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
    ${CERES_LIBRARIES}
    ${Boost_LIBRARIES}
    LiePlusPlus
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


add_executable(lie_example src/lie_plusplus.cpp)
target_include_directories(lie_example PRIVATE include)
target_link_libraries(lie_example ceres_nav ${CERES_LIBRARIES} ${Boost_LIBRARIES} LiePlusPlus)