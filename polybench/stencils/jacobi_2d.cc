#include <vector>

#include <common.h>
class Jacobi_2d;

using BENCH_DATA_TYPE = float;

void jacobi2d(
        celerity::distr_queue& queue,
        celerity::buffer<BENCH_DATA_TYPE, 2>& mat_a,
        celerity::buffer<BENCH_DATA_TYPE, 2>& mat_res,
        const size_t mat_size
        ){
        queue.submit([=](celerity::handler& cgh) {
            auto A = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(1,1));
            auto RES = mat_res.template get_access<cl::sycl::access::mode::write>(cgh, celerity::access::neighborhood<2>(1,1));

            cgh.parallel_for<class Jacobi_2d>(cl::sycl::range<2> (mat_size - 1, mat_size - 1), cl::sycl::id<2> {1,1}, [=](cl::sycl::item<2> item) {
                auto i = item[0];
                auto j = item[1];
                RES[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
            });
        });
}

class Jacobi_2d {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_res;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_res_buf;

public:
    Jacobi_2d(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_res = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);

        for (size_t i = 0; i < mat_size; i++)
            for (size_t j = 0; j < mat_size; j++){
                mat_a[(i*mat_size)+j] = ((BENCH_DATA_TYPE) i*(j+2) + 2) / mat_size;
                mat_res[(i*mat_size)+j] = ((BENCH_DATA_TYPE) i*(j+3) + 3) / mat_size;
            }

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_res_buf.initialize(mat_res.data(), cl::sycl::range<2>(mat_size, mat_size));
    }

    void run() {
        jacobi2d(QueueManager::getInstance(),
                 mat_a_buf.get(),
                 mat_res_buf.get(), mat_size);
    }

    static std::string getBenchmarkName() { return "Jacobi_2d"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_res_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));
            cgh.run([=, &verification_passed]() {
                for(size_t i = 1; i < mat_size -1; ++i)
                    for(size_t j = 1; j < mat_size -1; ++j) {
                        const float kernel_value = result[{i, j}];
                        const float host_value = 0.2 * (mat_a[(i*mat_size)+j] + mat_a[(i*mat_size)+(j-1)]
                                                        + mat_a[(i*mat_size)+(1+j)] + mat_a[((i+1)*mat_size)+j]
                                                        + mat_a[((i-1)*mat_size)+j]);
                        verification_passed = almost_equal(kernel_value, host_value, 1.f);
                    }
            });
        });
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);
    app.run< Jacobi_2d >();
}
