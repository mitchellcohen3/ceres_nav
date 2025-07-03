# find_package(Catch2 REQUIRED)
add_executable(test_factor_graph test_factor_graph.cpp)
target_link_libraries(test_factor_graph PRIVATE ${PROJECT_NAME} Catch2::Catch2WithMain)

add_executable(test_jacobians test_jacobians.cpp)
target_link_libraries(test_jacobians PRIVATE ${PROJECT_NAME} Catch2::Catch2WithMain)

add_executable(test_parameter_blocks test_parameter_blocks.cpp)
target_link_libraries(test_parameter_blocks PRIVATE ${PROJECT_NAME} Catch2::Catch2WithMain)

add_executable(test_state_collection test_state_collection.cpp)
target_link_libraries(test_state_collection PRIVATE ${PROJECT_NAME} Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(test_factor_graph)
catch_discover_tests(test_jacobians)
catch_discover_tests(test_parameter_blocks)
catch_discover_tests(test_state_collection)