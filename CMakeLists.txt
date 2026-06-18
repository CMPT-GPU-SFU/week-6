cmake_minimum_required(VERSION 3.18)

project(week6_tensor_cores LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to build for" FORCE)

add_executable(week6_tensor_cores
  src/main.cu
  src/wmma_fp16.cu
  src/wmma_tf32.cu)

target_include_directories(week6_tensor_cores PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_options(week6_tensor_cores PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(week6_tensor_cores PRIVATE CUDA::cublas)
