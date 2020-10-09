#include <cstdio>
#include <vector>

#include <common.h>
using BENCH_DATA_TYPE = float;

namespace values {
    constexpr BENCH_DATA_TYPE alpha = 32412;
    constexpr BENCH_DATA_TYPE beta = 2123;
} // namespace values



void gemm(celerity::distr_queue& queue,
        celerity::buffer<BENCH_DATA_TYPE, 2>& mat_a, celerity::buffer<BENCH_DATA_TYPE, 2>& mat_b,
        celerity::buffer<BENCH_DATA_TYPE, 2>& mat_res,
        const size_t mat_size){
    queue.submit([=](celerity::handler& cgh) {
        auto a = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
        auto b = mat_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
        auto res = mat_res.template get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::one_to_one<2>());
        cgh.parallel_for<class Gemm>(mat_res.get_range(), [=, n = mat_size](cl::sycl::item<2> item) {
            const auto i = item[0];
            const auto j = item[1];
            res[item] *= values::beta;
            for(size_t k = 0; k < n; k++) {
                res[item] += values::alpha * a[{i, k}] * b[{k, j}];
            }
        });
    });
}


class Gemm;

class Gemm {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_b;
    std::vector<BENCH_DATA_TYPE> mat_res;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_b_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_res_buf;

public:
    Gemm(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_b = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_res = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);

        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                mat_a[i * mat_size + j] = ((BENCH_DATA_TYPE)i * j) / mat_size;
                mat_b[i * mat_size + j] = ((BENCH_DATA_TYPE)i * j +1)  / mat_size;
                mat_res[i * mat_size + j] = ((BENCH_DATA_TYPE)i * j +2) / mat_size;
            }
        }

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_b_buf.initialize(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_res_buf.initialize(mat_res.data(), cl::sycl::range<2>(mat_size, mat_size));
    }

    void run() {
        gemm(QueueManager::getInstance(), mat_a_buf.get(), mat_b_buf.get(), mat_res_buf.get(),mat_size);
    }

    static std::string getBenchmarkName() { return "Gemm"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_res_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));
            cgh.run([=, &verification_passed]() {

                std::vector<BENCH_DATA_TYPE>test(mat_size * mat_size);
                std::vector<BENCH_DATA_TYPE>test_a(mat_size * mat_size);
                std::vector<BENCH_DATA_TYPE>test_b(mat_size * mat_size);

                for(size_t i = 0; i < mat_size; ++i)
                    for(size_t j = 0; j < mat_size; ++j){
                        test_a[i * mat_size + j] = ((BENCH_DATA_TYPE)i * j) / mat_size;
                        test_b[i * mat_size + j] = ((BENCH_DATA_TYPE)i * j +1)  / mat_size;
                        test[i*mat_size+j] = ((BENCH_DATA_TYPE)i * j +2) / mat_size;
                    }
                for(size_t i = 0; i < mat_size && verification_passed; ++i)
                    for(size_t j = 0; j < mat_size && verification_passed; ++j){

                        test[i*mat_size+j] *= values::beta;
                        for(size_t k = 0; k < mat_size; k++) {
                            test[i*mat_size+j] += values::alpha * test_a[i*mat_size + k] * test_b[k* mat_size + j];
                        }
                        verification_passed = almost_equal(result[{i, j}], test[(i*mat_size)+j], 1);

                    }
            });
        });
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Gemm >();
}
