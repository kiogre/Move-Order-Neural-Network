cmake_minimum_required(VERSION 3.14)
project(JellyFishInference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Definisci il percorso di ONNX Runtime estratto
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.17.1")

# Specifica dove trovare gli header (.h) e le librerie (.so)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Definisci l'eseguibile dal file test.cpp
add_executable(test test.cpp)

# Collega la libreria onnxruntime all'eseguibile
target_link_libraries(test PRIVATE onnxruntime)

# Imposta RPATH per permettere all'eseguibile di trovare la libreria .so in esecuzione senza usare LD_LIBRARY_PATH
set_target_properties(test PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)