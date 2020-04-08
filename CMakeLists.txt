cmake_minimum_required (VERSION 3.5)
project(celerity-bench)

#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release)
#endif()

find_package(celerity CONFIG REQUIRED)

include_directories(
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

function(add_benchmark source_dir target_name)
  add_executable(${target_name} ${source_dir}/${target_name}.cc)

  add_celerity_to_target(
    TARGET ${target_name}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}/${target_name}.cc
  )

  set_target_properties( ${target_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${source_dir}"
  )
endfunction(add_benchmark)
#add_benchmark(micro arith)
#add_benchmark(micro DRAM)
#add_benchmark(micro sf)
#add_benchmark(micro L2)
#add_benchmark(single-kernel vec_add)
#add_benchmark(single-kernel scalar_prod)
#add_benchmark(single-kernel mol_dyn)
#add_benchmark(single-kernel sobel)
add_benchmark(single-kernel sobel5)
add_benchmark(single-kernel sobel7)

#add_benchmark(runtime matmulchain)

