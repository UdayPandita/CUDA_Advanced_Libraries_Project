cmake_minimum_required(VERSION 3.10)
project(GPU_EdgeDetection LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures (adjust based on your GPU)
# Common values: 30 (Kepler), 35, 52 (Maxwell), 61 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 86, 89, 90
set(CUDA_ARCH "75" CACHE STRING "CUDA architecture (default: 75 for RTX)")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# CUDA 13.1 rejects newer MSVC by default; this flag allows local development.
if(MSVC)
    string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
endif()

# Add CUDA library
add_library(sobel_kernel STATIC
    src/sobel_kernel.cu
)

# Include directories
target_include_directories(sobel_kernel PUBLIC
    src
)

# Optional: Add executable for testing
# add_executable(edge_detection_test edge_detection.cpp)
# target_link_libraries(edge_detection_test sobel_kernel ${OpenCV_LIBS})

# Installation targets (optional)
install(TARGETS sobel_kernel DESTINATION lib)
install(FILES sobel_kernel.cu DESTINATION include)

# Print build information
message(STATUS "CUDA Architecture: sm_${CUDA_ARCH}")
