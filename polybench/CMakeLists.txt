set(stencil
        seidel_2d
        jacobi_1d
        jacobi_2d
#        fdtd_apml
)


set(linear-algebra
##        bicg
        2mm
        gemm
        2DConvolution
        gesummv
##        syrk
##        syr2k
        3DConvolution
#        3mm
##        atax
##        correlation
##        covariance
##        gramschmidt
##        mvt
##        fdtd2d
        )

foreach(exe IN LISTS stencil)
  add_benchmark(stencils "${exe}" _ "")
endforeach(exe)

foreach(exe IN LISTS linear-algebra)
  add_benchmark(linear-algebra "${exe}" _ "")
endforeach(exe)

add_benchmark(linear-algebra atax _ "")
add_benchmark(linear-algebra atax kernel1 "BENCH_KERNEL=1")
add_benchmark(linear-algebra atax kernel2 "BENCH_KERNEL=2")
add_benchmark(linear-algebra atax kernel3 "BENCH_KERNEL=3")

add_benchmark(linear-algebra bicg _ "")
add_benchmark(linear-algebra bicg kernel1 "BENCH_KERNEL=1")
add_benchmark(linear-algebra bicg kernel2 "BENCH_KERNEL=2")

add_benchmark(linear-algebra correlation _ "")
add_benchmark(linear-algebra correlation kernel1 "BENCH_KERNEL=1")
add_benchmark(linear-algebra correlation kernel2 "BENCH_KERNEL=2")
add_benchmark(linear-algebra correlation kernel3 "BENCH_KERNEL=3")
add_benchmark(linear-algebra correlation kernel4 "BENCH_KERNEL=4")
##add_benchmark(linear-algebra correlation kernel5 "BENCH_KERNEL=5")
##
add_benchmark(linear-algebra covariance _ "")
add_benchmark(linear-algebra covariance kernel1 "BENCH_KERNEL=1")
add_benchmark(linear-algebra covariance kernel2 "BENCH_KERNEL=2")
add_benchmark(linear-algebra covariance kernel3 "BENCH_KERNEL=3")
##
add_benchmark(linear-algebra fdtd2d _ "")
add_benchmark(linear-algebra fdtd2d kernel1 "BENCH_KERNEL=1")
add_benchmark(linear-algebra fdtd2d kernel2 "BENCH_KERNEL=2")
add_benchmark(linear-algebra fdtd2d kernel3 "BENCH_KERNEL=3")
add_benchmark(linear-algebra fdtd2d kernel4 "BENCH_KERNEL=4")
##
add_benchmark(linear-algebra gramschmidt _ "")
add_benchmark(linear-algebra gramschmidt kernel1 "BENCH_KERNEL=1")
add_benchmark(linear-algebra gramschmidt kernel2 "BENCH_KERNEL=2")
add_benchmark(linear-algebra gramschmidt kernel3 "BENCH_KERNEL=3")
#
add_benchmark(linear-algebra mvt _ "")
add_benchmark(linear-algebra mvt kernel1 "BENCH_KERNEL=1")
add_benchmark(linear-algebra mvt kernel2 "BENCH_KERNEL=2")

add_benchmark(linear-algebra syrk _ "")
add_benchmark(linear-algebra syrk kernel1 "BENCH_KERNEL=1")
add_benchmark(linear-algebra syrk kernel2 "BENCH_KERNEL=2")
#
add_benchmark(linear-algebra syr2k _ "")
add_benchmark(linear-algebra syr2k kernel1 "BENCH_KERNEL=1")
add_benchmark(linear-algebra syr2k kernel2 "BENCH_KERNEL=2")
