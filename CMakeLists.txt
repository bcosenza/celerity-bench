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

function(add_benchmark source_dir target_base_name suffix props)
  if ("${props}" MATCHES ".+")
    string(REPLACE ";" ";-D" dflags "-D${props}")
    set(target_name "${target_base_name}_${suffix}")
  else()
    set(target_name "${target_base_name}")
  endif()

  add_executable(${target_name} ${source_dir}/${target_base_name}.cc)

  add_celerity_to_target(
    TARGET ${target_name}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}/${target_base_name}.cc
  )

  set(compdefs COMPILE_DEFINITIONS BENCH_DATA_TYPE=float COMPILE DEFINITIONS BENCH_DIMS=1)
  set_target_properties( ${target_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${source_dir}"
    COMPILE_DEFINITIONS "${props}"
  )
#  target_compile_definitions(${target_name} PRIVATE "BENCH_DATA_TYPE=float" PRIVATE "BENCH_DIMS=1")

  add_custom_target("${target_name}_sfeatures" ALL
    /data/nico/local/bin/syclcc-clang ${dflags}
      --hipsycl-gpu-arch=sm_52
      -I${CMAKE_CURRENT_SOURCE_DIR}/include/
      -I${local_dir}/include/celerity/
      -I${local_dir}/include/celerity/vendor/
      -I/usr/include/mpi/
      -c -emit-llvm
      ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}/${target_base_name}.cc
    COMMAND
      ${features_script}
      -i ${target_base_name}-cuda-nvptx64-nvidia-cuda-sm_52.bc
      -o "${CMAKE_BINARY_DIR}/${source_dir}/${target_name}-cuda.sfeatures"
      -fs gpu
    COMMAND
      ${features_script}
      -i ${target_base_name}.bc
      -o "${CMAKE_BINARY_DIR}/${source_dir}/${target_name}.sfeatures"
      -fs gpu
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${target_name}_sfeatures.dir"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}/${target_base_name}.cc
  )
endfunction(add_benchmark)
#add_benchmark(micro arith int "BENCH_DATA_TYPE=int")
#add_benchmark(micro arith float "BENCH_DATA_TYPE=float")
#add_benchmark(micro arith double "BENCH_DATA_TYPE=double")
#add_benchmark(micro DRAM "float_1d" "BENCH_DATA_TYPE=float;BENCH_DIMS=1")
#add_benchmark(micro DRAM "float_2d" "BENCH_DATA_TYPE=float;BENCH_DIMS=2")
#add_benchmark(micro DRAM "float_3d" "BENCH_DATA_TYPE=float;BENCH_DIMS=3")
#add_benchmark(micro DRAM "double_1d" "BENCH_DATA_TYPE=double;BENCH_DIMS=1")
#add_benchmark(micro DRAM "double_2d" "BENCH_DATA_TYPE=double;BENCH_DIMS=2")
#add_benchmark(micro DRAM "double_3d" "BENCH_DATA_TYPE=double;BENCH_DIMS=3")
#add_benchmark(micro L2 int_1 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=1")
#add_benchmark(micro L2 int_2 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=2")
#add_benchmark(micro L2 int_4 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=4")
#add_benchmark(micro L2 int_8 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=8")
#add_benchmark(micro L2 int_16 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=16")
#add_benchmark(micro L2 float_1 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=1")
#add_benchmark(micro L2 float_2 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=2")
#add_benchmark(micro L2 float_4 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=4")
#add_benchmark(micro L2 float_8 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=8")
#add_benchmark(micro L2 float_16 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=16")
#add_benchmark(micro L2 double_1 "BENCH_DATA_TYPE=double;BENCH_COMP_ITERS=1")
#add_benchmark(micro L2 double_2 "BENCH_DATA_TYPE=double;BENCH_COMP_ITERS=2")
#add_benchmark(micro L2 double_4 "BENCH_DATA_TYPE=double;BENCH_COMP_ITERS=4")
#add_benchmark(micro L2 double_8 "BENCH_DATA_TYPE=double;BENCH_COMP_ITERS=8")
#add_benchmark(micro L2 double_16 "BENCH_DATA_TYPE=double;BENCH_COMP_ITERS=16")
#add_benchmark(micro sf float "BENCH_DATA_TYPE=float")
#add_benchmark(micro sf double "BENCH_DATA_TYPE=double")
#add_benchmark(micro multiple_buffers two_121 "BENCH_DATA_TYPE=int;BENCH_MODE=TWO_BUFFERS_121")
#add_benchmark(micro multiple_buffers four_121 "BENCH_DATA_TYPE=int;BENCH_MODE=FOUR_BUFFERS_121")
#add_benchmark(micro multiple_buffers six_121 "BENCH_DATA_TYPE=int;BENCH_MODE=SIX_BUFFERS_121")
#add_benchmark(micro multiple_buffers two_all "BENCH_DATA_TYPE=int;BENCH_MODE=TWO_BUFFERS_ALL")
#add_benchmark(micro multiple_buffers four_all "BENCH_DATA_TYPE=int;BENCH_MODE=FOUR_BUFFERS_ALL")
#add_benchmark(micro multiple_buffers six_all "BENCH_DATA_TYPE=int;BENCH_MODE=SIX_BUFFERS_ALL")
#add_benchmark(micro range_mappers one_to_one "BENCH_DATA_TYPE=int;BENCH_MAPPER=ONE_TO_ONE")
#add_benchmark(micro range_mappers neighbourhood "BENCH_DATA_TYPE=int;BENCH_MAPPER=NEIGHBOURHOOD")
#add_benchmark(micro range_mappers slice "BENCH_DATA_TYPE=int;BENCH_MAPPER=SLICE")
#add_benchmark(micro range_mappers fixed "BENCH_DATA_TYPE=int;BENCH_MAPPER=FIXED")
#add_benchmark(micro range_mappers all "BENCH_DATA_TYPE=int;BENCH_MAPPER=ALL")

add_benchmark(single-kernel matmul float "BENCH_DATA_TYPE=float")
add_benchmark(single-kernel median _ "")
add_benchmark(single-kernel mol_dyn _ "")
add_benchmark(single-kernel vec_add int "BENCH_DATA_TYPE=int")
add_benchmark(single-kernel vec_add long_long "BENCH_DATA_TYPE=long long")
add_benchmark(single-kernel vec_add float "BENCH_DATA_TYPE=float")
add_benchmark(single-kernel vec_add double "BENCH_DATA_TYPE=double")
add_benchmark(single-kernel sobel _ "")
add_benchmark(single-kernel sobel5 _ "")
add_benchmark(single-kernel sobel7 _ "")

# needs ndrange support in celerity:
#add_benchmark(single-kernel scalar_prod)

#add_benchmark(runtime matmulchain _ "")

