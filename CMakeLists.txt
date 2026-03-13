cmake_minimum_required(VERSION 3.10)
project(GPU_EdgeDetection CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Set CUDA architectures (adjust based on your GPU)
# Common values: 30 (Kepler), 35, 52 (Maxwell), 61 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 86, 89, 90
set(CUDA_ARCH "75" CACHE STRING "CUDA architecture (default: 75 for RTX)")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${CUDA_ARCH}")

# Add CUDA library
add_library(sobel_kernel STATIC
    sobel_kernel.cu
)

# Link OpenCV to CUDA kernel
target_link_libraries(sobel_kernel PUBLIC ${OpenCV_LIBS})

# Include directories
target_include_directories(sobel_kernel PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Optional: Add executable for testing
# add_executable(edge_detection_test edge_detection.cpp)
# target_link_libraries(edge_detection_test sobel_kernel ${OpenCV_LIBS})

# Installation targets (optional)
install(TARGETS sobel_kernel DESTINATION lib)
install(FILES sobel_kernel.cu DESTINATION include)

# Print build information
message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Architecture: sm_${CUDA_ARCH}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
