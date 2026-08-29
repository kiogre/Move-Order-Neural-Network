cmake_minimum_required(VERSION 3.14)
project(JellyFishInference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.20.0")
set(CONDA_NV_DIR "/home/koro/miniconda3/lib/python3.13/site-packages/nvidia")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE onnxruntime)

# Costruiamo la lista di tutti i percorsi NVIDIA di Conda
set(CUDA_RPATHS
    "${ONNXRUNTIME_DIR}/lib"
    "${CONDA_NV_DIR}/cublas/lib"
    "${CONDA_NV_DIR}/cufft/lib"
    "${CONDA_NV_DIR}/cudnn/lib"
    "${CONDA_NV_DIR}/cuda_runtime/lib"
    "${CONDA_NV_DIR}/cuda_nvrtc/lib"
)

# Unisce i percorsi con la sintassi corretta per il linker
string(JOIN ":" RPATH_JOINED ${CUDA_RPATHS})

# Applica l'RPATH sia per la fase di build sia per quella di installazione
set_target_properties(test PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "${RPATH_JOINED}"
    INSTALL_RPATH "${RPATH_JOINED}"
)