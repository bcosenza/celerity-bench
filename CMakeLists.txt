cmake_minimum_required (VERSION 3.5)
project(celerity-bench)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Celerity CONFIG REQUIRED)

set(features_script /p/project/celerity-rse/build/celerity-comp/feature_ext)
set(local_dir /p/project/celerity-rse/local)

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

#    cmake = f"cmake -DCMAKE_PREFIX_PATH={PREFIX_PATH} \
#              -DCELERITY_SYCL_IMPL=DPC++ \
#              -DCMAKE_BUILD_TYPE={BUILD_TYPE} \
#              -DCMAKE_C_COMPILER=/p/software/juwels/stages/2022/software/intel-compilers/2021.4.0/compiler/2021.4.0/linux/bin/clang \
#              -DCMAKE_CXX_COMPILER=/p/software/juwels/stages/2022/software/intel-compilers/2021.4.0/compiler/2021.4.0/linux/bin/clang++ \
#              {CODE_PATH}/celerity-bench/",
# /p/software/juwels/stages/2022/software/intel-compilers/2021.4.0/compiler/2021.4.0/linux/bin/clang++ -DCELERITY_FEATURE_LOCAL_ACCESSOR=1 -DCELERITY_FEATURE_SCALAR_REDUCTIONS=0 -DCELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS=1 -DSPDLOG_COMPILED_LIB -I/p/project/celerity-rse/code/celerity-bench/PUBLIC -I/p/project/celerity-rse/code/celerity-bench/include -isystem /p/project/celerity-rse/local/include/celerity -isystem /p/project/celerity-rse/local/include/celerity/vendor -isystem /p/software/juwels/stages/2022/software/impi/2021.4.0-intel-compilers-2021.4.0/mpi/2021.4.0/include -isystem /p/project/celerity-rse/local/include -O3 -fsycl -sycl-std=2020 -fsycl-targets= -DCELERITY_DPCPP=1 -fsycl-targets=spir64 -MD -MT CMakeFiles/matmulchain.dir/runtime/matmulchain.cc.o -MF CMakeFiles/matmulchain.dir/runtime/matmulchain.cc.o.d -o CMakeFiles/matmulchain.dir/runtime/matmulchain.cc.o -c /p/project/celerity-rse/code/celerity-bench/runtime/matmulchain.cc

  add_custom_target("${target_name}_sfeatures" ALL
    /p/software/juwels/stages/2022/software/intel-compilers/2021.4.0/compiler/2021.4.0/linux/bin/clang++ ${dflags}
      -DCELERITY_FEATURE_LOCAL_ACCESSOR=1 -DCELERITY_FEATURE_SCALAR_REDUCTIONS=0 -DCELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS=1 -DSPDLOG_COMPILED_LIB
      -I/p/project/celerity-rse/code/celerity-bench/PUBLIC
      -I/p/project/celerity-rse/code/celerity-bench/include
      -isystem /p/project/celerity-rse/local/include/celerity
      -isystem /p/project/celerity-rse/local/include/celerity/vendor
      -isystem /p/software/juwels/stages/2022/software/impi/2021.4.0-intel-compilers-2021.4.0/mpi/2021.4.0/include
      -isystem /p/project/celerity-rse/local/include
      -O3 -fsycl -sycl-std=2020 -fsycl-targets= -DCELERITY_DPCPP=1 -fsycl-targets=spir64 -fsycl-device-only
#      ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}
      -c -emit-llvm #-S
      ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}/${target_base_name}.cc
#    COMMAND
#      ${features_script}
#      -i ${target_base_name}-cuda-nvptx64-nvidia-cuda-sm_52.ll
#      -o "${CMAKE_BINARY_DIR}/${source_dir}/${target_name}-cuda.sfeatures"
#      -fs gpu -fe kofler
    COMMAND
      ${features_script}
      -i ${target_base_name}-sycl-spir64-unknown-unknown-sycldevice.bc
      -o "${CMAKE_BINARY_DIR}/${source_dir}/${target_name}.sfeatures"
      -fs gpu -fe kofler
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${target_name}_sfeatures.dir"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}/${target_base_name}.cc
  )
endfunction(add_benchmark)

#####add_subdirectory(polybench)

##add_benchmark(micro arith int_8 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=8")
#add_benchmark(micro arith int_16 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=16")
#add_benchmark(micro arith int_32 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=32")
#add_benchmark(micro arith int_64 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=64")
#add_benchmark(micro arith int_128 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=128")
#add_benchmark(micro arith int_256 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=256")
#add_benchmark(micro arith int_512 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=512")
##add_benchmark(micro arith int_1024 "BENCH_DATA_TYPE=int;BENCH_COMP_ITERS=1024")
##add_benchmark(micro arith float_8 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=8")
#add_benchmark(micro arith float_16 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=16")
#add_benchmark(micro arith float_32 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=32")
#add_benchmark(micro arith float_64 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=64")
#add_benchmark(micro arith float_128 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=128")
#add_benchmark(micro arith float_256 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=256")
#add_benchmark(micro arith float_512 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=512")
##add_benchmark(micro arith float_1024 "BENCH_DATA_TYPE=float;BENCH_COMP_ITERS=1024")
##add_benchmark(micro arith double "BENCH_DATA_TYPE=double")
#add_benchmark(micro DRAM float_1d "BENCH_DATA_TYPE=float;BENCH_DIMS=1")
#add_benchmark(micro DRAM float_2d "BENCH_DATA_TYPE=float;BENCH_DIMS=2")
#add_benchmark(micro DRAM float_3d "BENCH_DATA_TYPE=float;BENCH_DIMS=3")
##add_benchmark(micro DRAM double_1d "BENCH_DATA_TYPE=double;BENCH_DIMS=1")
##add_benchmark(micro DRAM double_2d "BENCH_DATA_TYPE=double;BENCH_DIMS=2")
##add_benchmark(micro DRAM double_3d "BENCH_DATA_TYPE=double;BENCH_DIMS=3")
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
##add_benchmark(micro L2 double_1 "BENCH_DATA_TYPE=double;BENCH_COMP_ITERS=1")
##add_benchmark(micro L2 double_2 "BENCH_DATA_TYPE=double;BENCH_COMP_ITERS=2")
##add_benchmark(micro L2 double_4 "BENCH_DATA_TYPE=double;BENCH_COMP_ITERS=4")
##add_benchmark(micro L2 double_8 "BENCH_DATA_TYPE=double;BENCH_COMP_ITERS=8")
##add_benchmark(micro L2 double_16 "BENCH_DATA_TYPE=double;BENCH_COMP_ITERS=16")
##add_benchmark(micro sf float "BENCH_DATA_TYPE=float")
##add_benchmark(micro sf double "BENCH_DATA_TYPE=double")
#add_benchmark(micro multiple_buffers two_121 "BENCH_DATA_TYPE=int; BENCH_BUFFERS=2;BENCH_MODE_121")
#add_benchmark(micro multiple_buffers four_121 "BENCH_DATA_TYPE=int;BENCH_BUFFERS=4;BENCH_MODE_121")
#add_benchmark(micro multiple_buffers six_121 "BENCH_DATA_TYPE=int; BENCH_BUFFERS=6;BENCH_MODE_121")
#add_benchmark(micro multiple_buffers two_all "BENCH_DATA_TYPE=int; BENCH_BUFFERS=2;BENCH_MODE_ALL")
#add_benchmark(micro multiple_buffers four_all "BENCH_DATA_TYPE=int;BENCH_BUFFERS=4;BENCH_MODE_ALL")
#add_benchmark(micro multiple_buffers six_all "BENCH_DATA_TYPE=int; BENCH_BUFFERS=6;BENCH_MODE_ALL")
#
#add_benchmark(micro range_mappers one_to_one "BENCH_DATA_TYPE=float;BENCH_MAPPER_ONE_TO_ONE")
#add_benchmark(micro range_mappers neighbourhood "BENCH_DATA_TYPE=float;BENCH_MAPPER_NEIGHBOURHOOD")
#add_benchmark(micro range_mappers slicex "BENCH_DATA_TYPE=float;BENCH_MAPPER_SLICE_X")
#add_benchmark(micro range_mappers slicey "BENCH_DATA_TYPE=float;BENCH_MAPPER_SLICE_Y")
#add_benchmark(micro range_mappers fixed "BENCH_DATA_TYPE=float;BENCH_MAPPER_FIXED")
#add_benchmark(micro range_mappers all "BENCH_DATA_TYPE=float;BENCH_MAPPER_ALL")
#
#add_benchmark(micro sequential_range_mappers one_to_one_all "BENCH_DATA_TYPE=int;BENCH_MAPPER_ONE_TO_ONE_ALL")

#add_benchmark(micro sequential_range_mappers one_to_one_all              "BENCH_DATA_TYPE=float;BENCH_MAPPER_ONE_TO_ONE_ALL")
#add_benchmark(micro sequential_range_mappers all_one_to_one              "BENCH_DATA_TYPE=float;BENCH_MAPPER_ALL_ONE_TO_ONE")
#add_benchmark(micro sequential_range_mappers one_to_one_neighbourhood    "BENCH_DATA_TYPE=float;BENCH_MAPPER_ONE_TO_ONE_NEIGHBOURHOOD")
#add_benchmark(micro sequential_range_mappers one_to_one_slicex           "BENCH_DATA_TYPE=float;BENCH_MAPPER_ONE_TO_ONE_SLICEX")
#add_benchmark(micro sequential_range_mappers one_to_one_slicey           "BENCH_DATA_TYPE=float;BENCH_MAPPER_ONE_TO_ONE_SLICEY")
#add_benchmark(micro sequential_range_mappers one_to_one_one_to_one       "BENCH_DATA_TYPE=float;BENCH_MAPPER_ONE_TO_ONE_ONE_TO_ONE")
#add_benchmark(micro sequential_range_mappers all_all                     "BENCH_DATA_TYPE=float;BENCH_MAPPER_ALL_ALL")
#add_benchmark(micro sequential_range_mappers neighbourhood_neighbourhood "BENCH_DATA_TYPE=float;BENCH_MAPPER_NEIGHBOURHOOD_NEIGHBOURHOOD")
#add_benchmark(micro sequential_range_mappers all_neighbourhood           "BENCH_DATA_TYPE=float;BENCH_MAPPER_ALL_NEIGHBOURHOOD")
#add_benchmark(micro sequential_range_mappers slicex_slicey               "BENCH_DATA_TYPE=float;BENCH_MAPPER_SLICEX_SLILCEY")
#add_benchmark(micro sequential_range_mappers slicey_slicex               "BENCH_DATA_TYPE=float;BENCH_MAPPER_SLICEY_SLILCEX")

#####add_benchmark(single-kernel matmul float "BENCH_DATA_TYPE=float")
#####add_benchmark(single-kernel sobel _ "")
#####add_benchmark(single-kernel sobel5 _ "")
#####add_benchmark(single-kernel sobel7 _ "")
#####add_benchmark(single-kernel median _ "")
#####add_benchmark(single-kernel mol_dyn _ "")
#####add_benchmark(single-kernel vec_add float "BENCH_DATA_TYPE=float")
#add_benchmark(single-kernel vec_add int "BENCH_DATA_TYPE=int")
#add_benchmark(single-kernel vec_add long_long "BENCH_DATA_TYPE=long long")
#add_benchmark(single-kernel vec_add double "BENCH_DATA_TYPE=double")

# needs ndrange support in celerity:
#add_benchmark(single-kernel scalar_prod)

add_benchmark(runtime matmulchain _ "")

#add_benchmark(single-kernel standalone _ "")
