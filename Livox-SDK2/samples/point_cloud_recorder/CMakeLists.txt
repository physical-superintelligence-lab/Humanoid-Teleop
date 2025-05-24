cmake_minimum_required(VERSION 3.0)

set(DEMO_NAME point_cloud_recorder)
add_executable(${DEMO_NAME} main.cpp)

find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

target_link_libraries(${DEMO_NAME}
    PUBLIC
    livox_lidar_sdk_static
    ${PCL_LIBRARIES}
)
