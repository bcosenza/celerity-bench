#include <cstdio>
#include <vector>

#include <common.h>

using BENCH_DATA_TYPE = float;
namespace values {
    constexpr BENCH_DATA_TYPE alpha = 123;
    constexpr BENCH_DATA_TYPE beta = 14512;
} // namespace values

class Syr2k;

class Syr2k {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_b;
    std::vector<BENCH_DATA_TYPE> mat_c;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_b_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_c_buf;

public:
    Syr2k(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_b = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_c = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);

        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                mat_a[(i * mat_size) + j] = ((BENCH_DATA_TYPE)i*j)/mat_size;
                mat_b[(i * mat_size) + j] = ((BENCH_DATA_TYPE)i*j)/mat_size;
                mat_c[(i * mat_size) + j] = ((BENCH_DATA_TYPE)i*j)/mat_size;
            }
        }
        auto range = celerity::range<2>(mat_size, mat_size);
        mat_a_buf.initialize(mat_a.data(), range);
        mat_b_buf.initialize(mat_b.data(), range);
        mat_c_buf.initialize(mat_c.data(), range);
    }

    void run() {
#if BENCH_KERNEL == 1 || !defined( BENCH_KERNEL )
        QueueManager::getInstance().submit([=](celerity::handler& cgh) {
            celerity::accessor c{mat_c_buf.get(), cgh, celerity::access::one_to_one{}, celerity::read_write};
            cgh.parallel_for<class Syr2k1>(cl::sycl::range<2>(mat_size, mat_size), [=](celerity::item<2> item)
            { c[item] *= values::beta; });
        });
#endif
#if BENCH_KERNEL == 2 || !defined( BENCH_KERNEL )
        QueueManager::getInstance().submit([=](celerity::handler& cgh) {
            celerity::accessor A{mat_a_buf.get(), cgh, celerity::access::all{}, celerity::read_only};
            celerity::accessor B{mat_b_buf.get(), cgh, celerity::access::all{}, celerity::read_only};
            celerity::accessor C{mat_b_buf.get(), cgh, celerity::access::one_to_one{}, celerity::read_write};

            cgh.parallel_for<class Syr2k2>(cl::sycl::range<2>(mat_size, mat_size), [=, n = mat_size](celerity::item<2> item) {
                const auto i = item[0];
                const auto j = item[1];

                for(size_t k = 0; k < n; ++k) {
                    C[item] += values::alpha * A[{i, k}] * B[{j, k}];
                    C[item] += values::alpha * B[{i, k}] * A[{j, k}];
                }
            });
        });
#endif
    }

    static std::string getBenchmarkName() { return "Syr2k"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        /*QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_c_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));
            cgh.run([=, &verification_passed]() {
                for(size_t i = 0; i < mat_size && verification_passed; ++i)
                    for(size_t j = 0; j < mat_size && verification_passed; ++j) {
                        auto test = mat_a[(i*mat_size)+j] * values::beta;
                        for(size_t k = 0; k < mat_size; ++k){
                            test += values::alpha * (mat_a[((i*mat_size)+k)] * mat_b[((j*mat_size)+k)]);
                            test += values::alpha * (mat_a[((i*mat_size)+k)] * mat_b[((j*mat_size)+k)]);
                        }
                        verification_passed = almost_equal(result[{i, j}], test, 5);
                    }
            });
        });*/
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Syr2k >();
}
