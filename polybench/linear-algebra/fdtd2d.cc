#include <cstdio>
#include <vector>

#include <common.h>

using BENCH_DATA_TYPE = float;
constexpr auto TMAX = 500;

class Fdtd2d;

void fdtd2d(celerity::distr_queue& queue,
           celerity::buffer<BENCH_DATA_TYPE, 2>& fict_buf,
           celerity::buffer<BENCH_DATA_TYPE, 2>& ex_buf,
           celerity::buffer<BENCH_DATA_TYPE, 2>& ey_buf,
           celerity::buffer<BENCH_DATA_TYPE, 2>& hz_buf,
           const size_t mat_size) {
    using namespace cl::sycl;
    using namespace celerity::access;
    for(size_t t = 0; t < TMAX; t++) {
#if KERNEL == 1 || !defined( KERNEL )
        queue.submit([=](celerity::handler& cgh) {
            auto fict = fict_buf.template get_access<access::mode::read>(cgh, celerity::access::fixed<2>({{t, 0}, {t, 0}}));
            auto ey = ey_buf.template get_access<access::mode::discard_write>(cgh, one_to_one<2>());
            cgh.parallel_for<class Fdtd2d1>(range<2>(1, mat_size), [=](item<2> item) { ey[item] = fict[{t, 0}]; });
        });
#endif
#if KERNEL == 2 || !defined( KERNEL )
        queue.submit([=](celerity::handler& cgh) {
            auto ey = ey_buf.template get_access<access::mode::read_write>(cgh, one_to_one<2>());
            auto hz = hz_buf.template get_access<access::mode::read>(cgh, neighborhood<2>(1, 1));
            cgh.parallel_for<class Fdtd2d2>(range<2>(mat_size - 1, mat_size), id<2>(1, 0), [=](item<2> item) {
                const auto i = item[0];
                const auto j = item[1];
                ey[item] = ey[item] - 0.5 * (hz[item] - hz[{(i - 1), j}]);
            });
        });
#endif
#if KERNEL == 3 || !defined( KERNEL )
        queue.submit([=](celerity::handler& cgh) {
            auto ex = ex_buf.template get_access<access::mode::read_write>(cgh, one_to_one<2>());
            auto hz = hz_buf.template get_access<access::mode::read>(cgh, neighborhood<2>(1, 1));
            cgh.parallel_for<class Fdtd2d3>(range<2>(mat_size, mat_size - 1), id<2>(0, 1), [=](item<2> item) {
                const auto i = item[0];
                const auto j = item[1];
                ex[item] = ex[item] - 0.5 * (hz[item] - hz[{i, (j - 1)}]);
            });
        });
#endif
#if KERNEL == 4 || !defined( KERNEL )
        queue.submit([=](celerity::handler& cgh) {
            auto ex = ex_buf.template get_access<access::mode::read>(cgh, neighborhood<2>(1, 1));
            auto ey = ey_buf.template get_access<access::mode::read>(cgh, neighborhood<2>(1, 1));
            auto hz = hz_buf.template get_access<access::mode::read_write>(cgh, one_to_one<2>());
            cgh.parallel_for<class Fdtd2d4>(hz_buf.get_range(), [=](item<2> item) {
                const auto i = item[0];
                const auto j = item[1];
                hz[item] = hz[item] - 0.7 * (ex[{i, (j + 1)}] - ex[item] + ey[{(i + 1), j}] - ey[item]);
            });
        });
#endif
    }

}


class Fdtd2d {
protected:
    std::vector<BENCH_DATA_TYPE> fict;
    std::vector<BENCH_DATA_TYPE> ex;
    std::vector<BENCH_DATA_TYPE> ey;
    std::vector<BENCH_DATA_TYPE> hz;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> fict_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> ex_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> ey_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> hz_buf;

public:
    Fdtd2d(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        fict = std::vector<BENCH_DATA_TYPE>(TMAX);
        ex = std::vector<BENCH_DATA_TYPE>(mat_size * (mat_size+1));
        ey = std::vector<BENCH_DATA_TYPE>((mat_size+1) * mat_size);
        hz = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        
        for(size_t i = 0; i < TMAX; i++)
            fict[i] = (BENCH_DATA_TYPE)i;

        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                ex[i * mat_size + j] = ((BENCH_DATA_TYPE)i * (j + 1) + 1) / mat_size;
                ey[i * mat_size + j] = ((BENCH_DATA_TYPE)(i - 1) * (j + 2) + 2) / mat_size;
                hz[i * mat_size + j] = ((BENCH_DATA_TYPE)(i - 9) * (j + 4) + 3) / mat_size;
            }
        }

        fict_buf.initialize(fict.data(), cl::sycl::range<2>(TMAX, 1));
        ex_buf.initialize(ex.data(), cl::sycl::range<2>(mat_size, (mat_size+1)));
        ey_buf.initialize(ey.data(), cl::sycl::range<2>(mat_size+1, mat_size));
        hz_buf.initialize(hz.data(), cl::sycl::range<2>(mat_size, mat_size));
    }

    void run() {
        fdtd2d(QueueManager::getInstance(),
              fict_buf.get(),ex_buf.get(),
              ey_buf.get(),hz_buf.get(),
              mat_size);
    }

    static std::string getBenchmarkName() { return "Fdtd2d"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        //todo
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Fdtd2d >();
}
