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
  add_benchmark(stencils "${exe}" _ "")
endforeach(exe)


foreach(exe IN LISTS linear-algebra)
  add_benchmark(linear-algebra "${exe}" _ "")
endforeach(exe)


