#include <cstdio>
#include <vector>

#include <common.h>

using BENCH_DATA_TYPE = float;
namespace values {
    constexpr BENCH_DATA_TYPE alpha = 123;
    constexpr BENCH_DATA_TYPE beta = 14512;
} // namespace values


void syrk(
        celerity::distr_queue& queue,
        celerity::buffer<BENCH_DATA_TYPE, 2>& mat_a,
        celerity::buffer<BENCH_DATA_TYPE, 2>& mat_res,
        const size_t mat_size
        ){
#if KERNEL == 1 || !defined( KERNEL )
    queue.submit([=](celerity::handler& cgh) {
        auto res = mat_res.template
                get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::one_to_one<2>());
        cgh.parallel_for<class Syrk1>(cl::sycl::range<2>(mat_size, mat_size), [=](cl::sycl::item<2> item)
        { res[item] *= values::beta; });
    });
#endif
#if KERNEL == 2 || !defined( KERNEL )
    QueueManager::getInstance().submit([=](celerity::handler& cgh) {
        auto A = mat_a.template
                get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
        auto res = mat_res.template
                get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::one_to_one<2>());
        cgh.parallel_for<class Syrk2>(cl::sycl::range<2>(mat_size, mat_size), [=, n = mat_size](cl::sycl::item<2> item) {
            const auto i = item[0];
            const auto j = item[1];

            for(size_t k = 0; k < n; ++k) {
                res[item] += values::alpha * (A[{i, k}] * A[{j, k}]);
            }
        });
    });
#endif
}

class Syrk;

class Syrk {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_res;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_res_buf;

public:
    Syrk(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_res = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);

        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                mat_a[(i * mat_size) + j] = ((BENCH_DATA_TYPE)i*j)/mat_size;
                mat_res[(i * mat_size) + j] = ((BENCH_DATA_TYPE)i*j)/mat_size;
            }
        }

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_res_buf.initialize(mat_res.data(),cl::sycl::range<2>(mat_size, mat_size));
    }

    void run() {
        syrk(QueueManager::getInstance(),
             mat_a_buf.get(), mat_res_buf.get(), mat_size);
    }

    static std::string getBenchmarkName() { return "Syrk"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_res_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));
            cgh.run([=, &verification_passed]() {
                for(size_t i = 0; i < mat_size && verification_passed; ++i)
                    for(size_t j = 0; j < mat_size && verification_passed; ++j) {
                        auto test = mat_a[(i*mat_size)+j] * values::beta;
                        for(size_t k = 0; k < mat_size; ++k)
                            test += values::alpha * (mat_a[((i*mat_size)+k)] * mat_a[((j*mat_size)+k)]);
                        verification_passed = almost_equal(result[{i, j}], test, 5);
                    }
            });
        });
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Syrk >();
}
