cmake_minimum_required(VERSION 3.14)
project(JellyFishInference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.20.0")

# Includi le cartelle degli header
include_directories(
    ${ONNXRUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/chess-library/include"
    "/home/koro/Scrivania/Move-Order-Neural-Network/MLChess/MLChess/C++"
)

link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE onnxruntime)

# RPATH configurato per le librerie NVIDIA di Conda
set(CONDA_NV_DIR "/home/koro/miniconda3/lib/python3.13/site-packages/nvidia")
set(CUDA_RPATHS
    "${ONNXRUNTIME_DIR}/lib"
    "${CONDA_NV_DIR}/cublas/lib"
    "${CONDA_NV_DIR}/cufft/lib"
    "${CONDA_NV_DIR}/cudnn/lib"
    "${CONDA_NV_DIR}/cuda_runtime/lib"
)
string(JOIN ":" RPATH_JOINED ${CUDA_RPATHS})

set_target_properties(test PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "${RPATH_JOINED}"
    INSTALL_RPATH "${RPATH_JOINED}"
)