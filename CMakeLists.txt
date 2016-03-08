cmake_minimum_required(VERSION 2.8)

project (nova)
set(CMAKE_BUILD_TYPE Release)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/mdp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pomdp
  )
# CMakeLists.txt to build hellocuda.cu
find_package(CUDA QUIET REQUIRED)
list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;-DVERBOSE")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Specify binary name and source file to build it from
cuda_add_library(
    nova SHARED
    src/mdp/algorithms/mdp_vi_cpu.cpp
    src/mdp/algorithms/ssp_rtdp_cpu.cpp
    src/mdp/algorithms/ssp_lao_star_cpu.cpp
    src/mdp/algorithms/mdp_vi_gpu.cu
    src/mdp/utilities/mdp_model_gpu.cu
    src/mdp/policies/mdp_value_function.cpp
    src/pomdp/algorithms/pomdp_pbvi_cpu.cpp
    src/pomdp/algorithms/pomdp_perseus_cpu.cpp
    src/pomdp/algorithms/pomdp_pbvi_gpu.cu  
    src/pomdp/policies/pomdp_alpha_vectors.cpp
    src/pomdp/utilities/pomdp_expand_cpu.cpp
    src/pomdp/utilities/pomdp_expand_gpu.cu
    src/pomdp/utilities/pomdp_model_gpu.cu
    src/pomdp/utilities/pomdp_sigma_cpu.cpp    
    )
install(TARGETS nova DESTINATION "${CMAKE_CURRENT_LIST_DIR}/lib")