cmake_minimum_required(VERSION 3.15)
project(SimpleTrackCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS /usr/include /usr/local/include /opt/onnxruntime/include)
find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS /usr/lib /usr/local/lib /opt/onnxruntime/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime library not found. Install onnxruntime-dev or set ONNXRUNTIME_INCLUDE_DIR/ONNXRUNTIME_LIB.")
endif()

file(GLOB SIMPLETRACK_SOURCES
    "mot_3d/association.cpp"
    "mot_3d/detections.cpp"
    "mot_3d/frame_data.cpp"
    "mot_3d/life/hit_manager.cpp"
    "mot_3d/mot.cpp"
    "mot_3d/motion_model/kalman_filter.cpp"
    "mot_3d/redundancy/redundancy.cpp"
    "mot_3d/tracklet/tracklet.cpp"
    "mot_3d/update_info_data.cpp"
    "mot_3d/utils/data_utils.cpp"
    "mot_3d/utils/geometry.cpp"
    "mot_3d/preprocessing/bbox_coarse_hash.cpp"
    "mot_3d/preprocessing/nms.cpp"
    "mot_3d/visualization/visualizer2d.cpp"
)

add_library(simpletrack STATIC ${SIMPLETRACK_SOURCES})
target_link_libraries(simpletrack
    PUBLIC
        Eigen3::Eigen
)

target_include_directories(simpletrack
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(TARGET Boost::headers)
    target_link_libraries(simpletrack PUBLIC Boost::headers)
else()
    target_include_directories(simpletrack PUBLIC ${Boost_INCLUDE_DIRS})
endif()

add_library(SimpleTrack::simpletrack ALIAS simpletrack)

add_executable(inference_sfa3d_at128_onnx tools/inference_sfa3d_at128_onnx.cpp)
target_include_directories(inference_sfa3d_at128_onnx
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(inference_sfa3d_at128_onnx
    PRIVATE
        SimpleTrack::simpletrack
        yaml-cpp
        ${ONNXRUNTIME_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
)
