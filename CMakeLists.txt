cmake_minimum_required(VERSION 3.15)
project(HailoPerceptionProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(SIMPLETRACK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTrack)

find_package(HailoRT REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs highgui)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

set(SIMPLETRACK_SOURCES
    ${SIMPLETRACK_DIR}/mot_3d/association.cpp
    ${SIMPLETRACK_DIR}/mot_3d/detections.cpp
    ${SIMPLETRACK_DIR}/mot_3d/frame_data.cpp
    ${SIMPLETRACK_DIR}/mot_3d/life/hit_manager.cpp
    ${SIMPLETRACK_DIR}/mot_3d/mot.cpp
    ${SIMPLETRACK_DIR}/mot_3d/motion_model/kalman_filter.cpp
    ${SIMPLETRACK_DIR}/mot_3d/redundancy/redundancy.cpp
    ${SIMPLETRACK_DIR}/mot_3d/tracklet/tracklet.cpp
    ${SIMPLETRACK_DIR}/mot_3d/update_info_data.cpp
    ${SIMPLETRACK_DIR}/mot_3d/utils/data_utils.cpp
    ${SIMPLETRACK_DIR}/mot_3d/utils/geometry.cpp
    ${SIMPLETRACK_DIR}/mot_3d/preprocessing/bbox_coarse_hash.cpp
    ${SIMPLETRACK_DIR}/mot_3d/preprocessing/nms.cpp
    ${SIMPLETRACK_DIR}/mot_3d/visualization/visualizer2d.cpp
)

add_library(simpletrack_core STATIC ${SIMPLETRACK_SOURCES})
target_include_directories(simpletrack_core
    PUBLIC
        ${SIMPLETRACK_DIR}/include
)
target_link_libraries(simpletrack_core
    PUBLIC
        Eigen3::Eigen
)
if(TARGET Boost::headers)
    target_link_libraries(simpletrack_core PUBLIC Boost::headers)
else()
    target_include_directories(simpletrack_core PUBLIC ${Boost_INCLUDE_DIRS})
endif()
target_link_libraries(simpletrack_core PRIVATE ${OpenCV_LIBS})

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/hailo_inference.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${SIMPLETRACK_DIR}/include
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        HailoRT::libhailort
        simpletrack_core
        yaml-cpp
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        Threads::Threads
)
