include_directories(common)

set(exes
  2DConvolution
  2mm
  3DConvolution
  3mm
  atax
  bicg
  correlation
  covariance
  fdtd2d
  gemm
  gesummv
  gramschmidt
  mvt
  syr2k
  syrk
)

foreach(exe IN LISTS exes)
  add_executable(${exe}_Celerity ${exe}/Celerity/${exe}.cpp)
  add_celerity_to_target(TARGET ${exe}_Celerity SOURCES ${exe}/Celerity/${exe}.cpp)
  target_compile_options(${exe}_Celerity PRIVATE -fdiagnostics-color=always -Wall)
endforeach(exe)
