cmake_minimum_required(VERSION 2.8.3)

project(nova)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

set(MODULE_NAME "nova")
set(LIB_NAME ${MODULE_NAME})
set(DIR_NAME ${MODULE_NAME})
set(EXPORT ${MODULE_NAME})
set(EXPORT_NAME "${EXPORT}_Export")
set(EXPORT_FILE "${EXPORT_NAME}.h")
string(TOUPPER "${EXPORT}_BUILD_DLL" EXPORT_DEFINE )

set(SOURCES
#    src/pomdp/utilities/pomdp_expand_gpu.cu
#    src/pomdp/utilities/pomdp_model_gpu.cu
    src/pomdp/utilities/pomdp_model_cpu.cpp
    src/pomdp/utilities/pomdp_sigma_cpu.cpp
    src/pomdp/utilities/pomdp_expand_cpu.cpp
#    src/pomdp/algorithms/pomdp_pbvi_gpu.cu
    src/pomdp/algorithms/pomdp_perseus_cpu.cpp
    src/pomdp/algorithms/pomdp_pbvi_cpu.cpp
    src/pomdp/policies/pomdp_alpha_vectors.cpp
#    src/mdp/utilities/mdp_model_gpu.cu
    src/mdp/utilities/mdp_model_cpu.cpp
    src/mdp/algorithms/ssp_rtdp_cpu.cpp
#    src/mdp/algorithms/mdp_vi_gpu.cu
    src/mdp/algorithms/ssp_lao_star_cpu.cpp
    src/mdp/algorithms/mdp_vi_cpu.cpp
    src/mdp/policies/mdp_value_function.cpp
)

set(HEADERS
    include/nova/constants.h
    include/nova/error_codes.h
    include/nova/pomdp/utilities/pomdp_model_cpu.h
    include/nova/pomdp/utilities/pomdp_expand_gpu.h
    include/nova/pomdp/utilities/pomdp_sigma_cpu.h
    include/nova/pomdp/utilities/pomdp_expand_cpu.h
    include/nova/pomdp/utilities/pomdp_model_gpu.h
    include/nova/pomdp/algorithms/pomdp_pbvi_cpu.h
    include/nova/pomdp/algorithms/pomdp_pbvi_gpu.h
    include/nova/pomdp/algorithms/pomdp_perseus_cpu.h
    include/nova/pomdp/policies/pomdp_alpha_vectors.h
    include/nova/pomdp/pomdp.h
    include/nova/mdp/utilities/mdp_model_gpu.h
    include/nova/mdp/utilities/mdp_model_cpu.h
    include/nova/mdp/algorithms/mdp_vi_gpu.h
    include/nova/mdp/algorithms/ssp_lao_star_cpu.h
    include/nova/mdp/algorithms/mdp_vi_cpu.h
    include/nova/mdp/algorithms/ssp_rtdp_cpu.h
    include/nova/mdp/policies/mdp_value_function.h
    include/nova/mdp/mdp.h
)

include_directories(
  include
)

add_library(${LIB_NAME}
  ${HEADERS}
  ${SOURCES}
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${LIB_NAME}
)

