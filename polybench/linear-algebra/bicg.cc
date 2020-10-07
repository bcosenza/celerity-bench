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

    queue.submit([=](celerity::handler& cgh) {
        auto A = mat_a.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
        auto r = mat_r.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
        auto s = mat_s.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<2>());

        cgh.parallel_for<class Bicg2>(mat_s.get_range(), [=, NX_ = mat_size](cl::sycl::item<2> item) {
            const auto j = item[0];

            BENCH_DATA_TYPE result = 0;
            for(size_t i = 0; i < NX_; i++) {
                result += r[{i, 0}] * A[{i, j}];
            }
            s[item] = result;
        });
    });

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
        mat_s = std::vector<BENCH_DATA_TYPE>(mat_size);
        mat_p = std::vector<BENCH_DATA_TYPE>(mat_size);
        mat_q = std::vector<BENCH_DATA_TYPE>(mat_size);

        for(size_t i = 0; i < mat_size; ++i) {
            mat_r[i] = i * M_PI;
            mat_p[i] = i * M_PI;
            for(size_t j = 0; j < mat_size; ++j) {
                mat_a[i * mat_size + j] = (i * j)/mat_size;

            }
        }

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_r_buf.initialize(mat_r.data(), cl::sycl::range<2>(mat_size, 1));
        mat_p_buf.initialize(mat_p.data(), cl::sycl::range<2>(mat_size, 1));
        mat_s_buf.initialize(cl::sycl::range<2>(mat_size, 1));
        mat_q_buf.initialize(cl::sycl::range<2>(mat_size, 1));
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
        //todo
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Bicg >();
}
