#include <cstdio>
#include <vector>

#include <common.h>

// Performs matrix multiply
using BENCH_DATA_TYPE = float;

class mm2;

void multiply(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& mat_a, celerity::buffer<BENCH_DATA_TYPE, 2>& mat_b,
              celerity::buffer<BENCH_DATA_TYPE, 2>& mat_c, const size_t mat_size) {
    queue.submit([=](celerity::handler& cgh) {
        auto a = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
        auto b = mat_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
        auto c = mat_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

        cgh.parallel_for<class Matmul>(cl::sycl::range<2>(mat_size, mat_size), [=](cl::sycl::item<2> item) {
            auto sum = 0.f;
            for(size_t k = 0; k < mat_size; ++k) {
                const auto a_ik = a[{item[0], k}];
                const auto b_kj = b[{k, item[1]}];
                sum += a_ik * b_kj;
            }
            c[item] = sum;
        });
    });
}


class mm2 {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_b;
    std::vector<BENCH_DATA_TYPE> mat_c;
    std::vector<BENCH_DATA_TYPE> mat_d;
    std::vector<BENCH_DATA_TYPE> mat_res;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_b_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_c_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_d_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_res_buf;

public:
    mm2(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_b = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_c = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size, 0);
        mat_d = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_res = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size, 0);

        // Initialize matrices to the identity
        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                mat_a[i * mat_size + j] = i == j;
                mat_b[i * mat_size + j] = i == j;
                mat_d[i * mat_size + j] = i == j;
            }
        }

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_b_buf.initialize(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_c_buf.initialize(mat_c.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_d_buf.initialize(mat_d.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_res_buf.initialize(cl::sycl::range<2>(mat_size, mat_size));
    }

    void run() {
        multiply(QueueManager::getInstance(), mat_a_buf.get(), mat_b_buf.get(),mat_c_buf.get(),mat_size);
        multiply(QueueManager::getInstance(), mat_c_buf.get(),mat_d_buf.get(), mat_res_buf.get(), mat_size);
    }

    static std::string getBenchmarkName() { return "mm2"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_res_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));
            cgh.run([=, &verification_passed]() {
                for(size_t i = 0; i < mat_size && verification_passed; ++i)
                    for(size_t j = 0; j < mat_size && verification_passed; ++j)
                        verification_passed = result[{i, j}] == (i == j);
            });
        });
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< mm2 >();
}
