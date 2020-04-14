cmake_minimum_required (VERSION 3.5)
project(celerity-bench)

#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release)
#endif()

find_package(celerity CONFIG REQUIRED)

set(features_script /data/nico/build/celerity-comp/features)
set(local_dir /data/nico/local)

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

  add_custom_target("${target_name}_sfeatures" ALL
    /data/nico/local/bin/syclcc-clang
      --hipsycl-gpu-arch=sm_52
      -I${CMAKE_CURRENT_SOURCE_DIR}/include/
      -I${local_dir}/include/celerity/
      -I${local_dir}/include/celerity/vendor/
      -I/usr/include/mpi/
      -c -emit-llvm
      ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}/${target_name}.cc
    COMMAND
      ${features_script}
      -i ${target_name}-cuda-nvptx64-nvidia-cuda-sm_52.bc
      -o "${CMAKE_BINARY_DIR}/${source_dir}/${target_name}-cuda.sfeatures"
      -fs gpu
    COMMAND
      ${features_script}
      -i ${target_name}.bc
      -o "${CMAKE_BINARY_DIR}/${source_dir}/${target_name}.sfeatures"
      -fs gpu
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${target_name}_sfeatures.dir"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}/${target_name}.cc
  )
endfunction(add_benchmark)
#add_benchmark(micro arith)
#add_benchmark(micro DRAM)
#add_benchmark(micro sf)
#add_benchmark(micro L2)
#add_benchmark(micro range_mappers)
add_benchmark(micro buffers)

#add_benchmark(single-kernel vec_add)
#add_benchmark(single-kernel scalar_prod)
#add_benchmark(single-kernel mol_dyn)
#add_benchmark(single-kernel sobel)
#add_benchmark(single-kernel sobel5)
#add_benchmark(single-kernel sobel7)
#add_benchmark(single-kernel matmul)

#add_benchmark(runtime matmulchain)

