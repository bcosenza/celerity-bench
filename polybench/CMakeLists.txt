set(stencil
        seidel_2d
        jacobi_1d
        jacobi_2d
        fdtd_apml
)


set(linear-algebra
        bicg
        2mm
        gemm
        2DConvolution
        gesummv
        syrk
        syr2k
        3DConvolution
        3mm
        atax
        correlation
        covariance
        gramschmidt
        mvt
        fdtd2d
        )

foreach(exe IN LISTS stencil)
  add_executable(${exe}_Celerity stencils/${exe}.cpp)
  add_celerity_to_target(TARGET ${exe}_Celerity SOURCES stencils/${exe}.cpp)
  target_compile_options(${exe}_Celerity PRIVATE -fdiagnostics-color=always -Wall)
endforeach(exe)



foreach(exe IN LISTS linear-algebra)
  add_executable(${exe}_Celerity linear-algebra/${exe}.cc)
  add_celerity_to_target(TARGET ${exe}_Celerity SOURCES linear-algebra/${exe}.cc)
  target_compile_options(${exe}_Celerity PRIVATE -fdiagnostics-color=always -Wall)
endforeach(exe)
