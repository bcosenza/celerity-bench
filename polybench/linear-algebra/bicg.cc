#include <cstdio>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <common.h>

using BENCH_DATA_TYPE = float;

class Bicg;

void bicg(celerity::distr_queue& queue,
          celerity::buffer<BENCH_DATA_TYPE, 2>& mat_a,
          celerity::buffer<BENCH_DATA_TYPE, 2>& mat_r,
          celerity::buffer<BENCH_DATA_TYPE, 2>& mat_s,
          celerity::buffer<BENCH_DATA_TYPE, 2>& mat_p,
          celerity::buffer<BENCH_DATA_TYPE, 2>& mat_q,
          const size_t mat_size) {

#if KERNEL == 1 || !defined( KERNEL )
    queue.submit([=](celerity::handler& cgh) {
        auto A = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
        auto p = mat_p.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
        auto q = mat_q.template get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<2>());

        cgh.parallel_for<class Bicg1>(mat_q.get_range(), [=, NY_ = mat_size](cl::sycl::item<2> item) {
            const auto i = item[0];

            BENCH_DATA_TYPE result = 0;
            for(size_t j = 0; j < NY_; j++) {
                result += A[{i, j}] * p[{j, 0}];
            }
            q[item] = result;
        });
    });
#endif
#if KERNEL == 2 || !defined( KERNEL )
    queue.submit([=](celerity::handler& cgh) {
        auto A = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
        auto r = mat_r.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
        auto s = mat_s.template get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<2>());

        cgh.parallel_for<class Bicg2>(mat_s.get_range(), [=, NX_ = mat_size](cl::sycl::item<2> item) {
            const auto j = item[0];

            BENCH_DATA_TYPE result = 0;
            for(size_t i = 0; i < NX_; i++) {
                result += r[{i, 0}] * A[{j, i}];
            }
            s[item] = result;
        });
    });
#endif
}


class Bicg {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_r;
    std::vector<BENCH_DATA_TYPE> mat_s;
    std::vector<BENCH_DATA_TYPE> mat_p;
    std::vector<BENCH_DATA_TYPE> mat_q;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_r_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_s_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_p_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_q_buf;

public:
    Bicg(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_r = std::vector<BENCH_DATA_TYPE>(mat_size);
        mat_s = std::vector<BENCH_DATA_TYPE>(mat_size,0.f);
        mat_p = std::vector<BENCH_DATA_TYPE>(mat_size);
        mat_q = std::vector<BENCH_DATA_TYPE>(mat_size,0.f);

        for(size_t i = 0; i < mat_size; ++i) {
            mat_r[i] = i * M_PI;
            mat_p[i] = i * M_PI;
            for(size_t j = 0; j < mat_size; ++j) {
                mat_a[i * mat_size + j] = ((BENCH_DATA_TYPE)i * j)/mat_size;

            }
        }

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_r_buf.initialize(mat_r.data(), cl::sycl::range<2>(mat_size, 1));
        mat_p_buf.initialize(mat_p.data(), cl::sycl::range<2>(mat_size, 1));
        mat_s_buf.initialize(mat_s.data(),cl::sycl::range<2>(mat_size, 1));
        mat_q_buf.initialize(mat_q.data(),cl::sycl::range<2>(mat_size, 1));
    }

    void run() {
        bicg(QueueManager::getInstance(),
             mat_a_buf.get(), mat_r_buf.get(),
             mat_s_buf.get(), mat_p_buf.get(),
             mat_q_buf.get(), mat_size);
    }

    static std::string getBenchmarkName() { return "Bicg"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto ress = mat_s_buf.get().template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, 1));
            auto resq = mat_q_buf.get().template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, 1));
            cgh.run([=, &verification_passed]() {
                std::vector<BENCH_DATA_TYPE> test_q(mat_size,0.f);
                std::vector<BENCH_DATA_TYPE> test_s(mat_size,0.f);
                for (int i = 0; i < mat_size; i++) {
                    for (int j = 0; j < mat_size; j++) {
                        test_s[j] = test_s[j] + mat_r[i] * mat_a[i * mat_size + j];
                        test_q[i] = test_q[i] + mat_a[i * mat_size + j] * mat_p[j];
                    }
                }
                for (int i=0;i<test_s.size() && verification_passed;i++) {
                    verification_passed =
                            almost_equal(ress[i][0], test_s[i], 3);
                }
                for (int i=0;i<test_q.size() && verification_passed;i++) {
                    verification_passed =
                            almost_equal(resq[i][0], test_q[i], 3);
                }
            });
        });
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);
    app.run< Bicg >();
}
