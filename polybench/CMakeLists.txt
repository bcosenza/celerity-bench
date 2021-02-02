set(stencil
        seidel_2d
        jacobi_1d
        jacobi_2d
        fdtd_apml
)


set(linear-algebra
#        bicg
        2mm
        gemm
        2DConvolution
        gesummv
#        syrk
#        syr2k
        3DConvolution
        3mm
#        atax
#        correlation
#        covariance
#        gramschmidt
#        mvt
#        fdtd2d
        )

foreach(exe IN LISTS stencil)
  add_benchmark(stencils "${exe}" _ "")
endforeach(exe)

add_benchmark(linear-algebra atax _ "")
add_benchmark(linear-algebra atax kernel1 "KERNEL=1")
add_benchmark(linear-algebra atax kernel2 "KERNEL=2")
add_benchmark(linear-algebra atax kernel3 "KERNEL=3")

add_benchmark(linear-algebra bicg _ "")
add_benchmark(linear-algebra bicg kernel1 "KERNEL=1")
add_benchmark(linear-algebra bicg kernel2 "KERNEL=2")

add_benchmark(linear-algebra correlation _ "")
add_benchmark(linear-algebra correlation kernel1 "KERNEL=1")
add_benchmark(linear-algebra correlation kernel2 "KERNEL=2")
add_benchmark(linear-algebra correlation kernel3 "KERNEL=3")
add_benchmark(linear-algebra correlation kernel4 "KERNEL=4")
add_benchmark(linear-algebra correlation kernel5 "KERNEL=5")

add_benchmark(linear-algebra covariance _ "")
add_benchmark(linear-algebra covariance kernel1 "KERNEL=1")
add_benchmark(linear-algebra covariance kernel2 "KERNEL=2")
add_benchmark(linear-algebra covariance kernel3 "KERNEL=3")

add_benchmark(linear-algebra fdtd2d _ "")
add_benchmark(linear-algebra fdtd2d kernel1 "KERNEL=1")
add_benchmark(linear-algebra fdtd2d kernel2 "KERNEL=2")
add_benchmark(linear-algebra fdtd2d kernel3 "KERNEL=3")
add_benchmark(linear-algebra fdtd2d kernel4 "KERNEL=4")

add_benchmark(linear-algebra gramschmidt _ "")
add_benchmark(linear-algebra gramschmidt kernel1 "KERNEL=1")
add_benchmark(linear-algebra gramschmidt kernel2 "KERNEL=2")
add_benchmark(linear-algebra gramschmidt kernel3 "KERNEL=3")

add_benchmark(linear-algebra mvt _ "")
add_benchmark(linear-algebra mvt kernel1 "KERNEL=1")
add_benchmark(linear-algebra mvt kernel2 "KERNEL=2")

add_benchmark(linear-algebra syrk _ "")
add_benchmark(linear-algebra syrk kernel1 "KERNEL=1")
add_benchmark(linear-algebra syrk kernel2 "KERNEL=2")

add_benchmark(linear-algebra syr2k _ "")
# add_benchmark(linear-algebra syr2k kernel1 "KERNEL=1") # is equivalent to syrk_kernel1
add_benchmark(linear-algebra syr2k kernel2 "KERNEL=2")
