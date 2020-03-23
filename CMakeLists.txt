cmake_minimum_required (VERSION 3.5)
project(celerity-bench)

#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release)
#endif()

find_package(celerity REQUIRED)

find_package(Boost 1.65.0 REQUIRED)
find_package(MPI 2.0 REQUIRED)
find_package(Threads REQUIRED)

include_directories(
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  /data/nico/local/include/celerity/celerity/
  /data/nico/local/include/celerity/vendor/
)

function(add_benchmark target_name source_dir)
  add_executable(${target_name} ${source_dir}/${target_name}.cc)

  target_link_libraries(
    ${target_name}
    PUBLIC
    celerity_runtime
    hipSYCL_cuda
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
  )

  add_sycl_to_target(
    TARGET ${target_name}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}/${target_name}.cc
  )

  set_target_properties( ${target_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${source_dir}"
  )
endfunction(add_benchmark)

#add_benchmark(matmul single-kernel)
add_benchmark(matmulchain runtime)

