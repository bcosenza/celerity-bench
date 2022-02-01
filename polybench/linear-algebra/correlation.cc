#include <cstdio>
#include <vector>

#include <common.h>
#define FLOAT_N 3214212.01
#define EPS 0.005

#define sqrt_of_array_cell(x, j) sqrt(x[j])

using BENCH_DATA_TYPE = float;

class Correlation;

void correlation(celerity::distr_queue queue,
                 celerity::buffer<BENCH_DATA_TYPE, 2> d,
                 celerity::buffer<BENCH_DATA_TYPE, 2> m,
                 celerity::buffer<BENCH_DATA_TYPE, 2> sd,
                 celerity::buffer<BENCH_DATA_TYPE, 2> sym,
                 const size_t mat_size) {
    using namespace cl::sycl;
#if BENCH_KERNEL == 1 || !defined( BENCH_KERNEL )
    queue.submit([=](celerity::handler& cgh) {
        celerity::accessor data{d, cgh, celerity::access::slice<2>(0), celerity::read_only};
        celerity::accessor mean{m, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
        cgh.parallel_for<class Correlation1>(range<2>(mat_size, 1), id<2>(1, 0), [=, N_ = mat_size](celerity::item<2> item) {
            const auto j = item[0];

            BENCH_DATA_TYPE result = 0;
            for(size_t i = 1; i <= N_; i++) {
                result += data[{i, j}];
            }
            mean[item] = result / ((BENCH_DATA_TYPE)FLOAT_N);
        });
    });
#endif
#if BENCH_KERNEL == 2 || !defined( BENCH_KERNEL )
    queue.submit([=](celerity::handler& cgh) {
        celerity::accessor data{d, cgh, celerity::access::slice<2>(0), celerity::read_only};
        celerity::accessor mean{m, cgh, celerity::access::one_to_one{}, celerity::read_only};
        celerity::accessor stddev{sd, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
        cgh.parallel_for<class Correlation2>(range<2>(mat_size, 1), id<2>(1, 0), [=, N_ = mat_size](celerity::item<2> item) {
            const auto j = item[0];

            BENCH_DATA_TYPE result = 0;
            for(size_t i = 1; i <= N_; i++) {
                result += (data[{i, j}] - mean[item]) * (data[{i, j}] - mean[item]);
            }
            result /= FLOAT_N;
            result = sqrt(result);

            stddev[item] = result <= EPS ? 1.0 : result;
        });
    });
#endif
#if BENCH_KERNEL == 3 || !defined( BENCH_KERNEL )
    queue.submit([=](celerity::handler& cgh) {
        celerity::accessor data{d, cgh, celerity::access::one_to_one{}, celerity::read_write};
        celerity::accessor mean{m, cgh, celerity::access::all{}, celerity::read_only};
        celerity::accessor stddev{sd, cgh, celerity::access::all{}, celerity::read_only};
        cgh.parallel_for<class Correlation3>(range<2>(mat_size, mat_size), id<2>(1, 1), [=](celerity::item<2> item) {
            const auto j = item[1];

            auto result = data[item];
            result -= mean[{j, 0}];
            result /= sqrt(FLOAT_N);
            result /= stddev[{j, 0}];

            data[item] = result;
        });
    });

#endif
#if BENCH_KERNEL == 4 || !defined( BENCH_KERNEL )
    queue.submit([=](celerity::handler& cgh) {
        celerity::accessor data{d, cgh, celerity::access::all{}, celerity::read_only};
        celerity::accessor symmat{sym, cgh, celerity::access::slice<2>(1), celerity::write_only, celerity::no_init};
        //celerity::accessor symmat2{sym, cgh, celerity::access::slice<2>(0), celerity::write_only, celerity::no_init};
        cgh.parallel_for<class Correlation4>(range<2>(mat_size - 1, 1), id<2>(1, 0), [=, M_ = mat_size, N_ = mat_size](celerity::item<2> item) {
            const auto j1 = item[0];

            symmat[{j1, j1}] = 1.0;

            for(size_t j2 = j1 + 1; j2 <= M_; j2++) {
                BENCH_DATA_TYPE result = 0.0;
                for(size_t i = 1; i <= N_; i++) {
                    result += data[{i, j1}] * data[{i, j2}];
                }

                symmat[{j1, j2}] = result;
                // symmat2[{j2, j1}] = result; // TODO this is currently not supported by celerity
            }
        });
    });
#endif
//#if BENCH_KERNEL == 5 || !defined( BENCH_KERNEL )
//    queue.submit([=](celerity::handler& cgh) {
//        celerity::accessor symmat{sym, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
//        cgh.parallel_for<class Correlation5>(celerity::range<2>(1, 1), id<2>(mat_size, mat_size), [=](celerity::item<2> item) { symmat[item] = 1.0; });
//    });
//#endif
}


class Correlation {
protected:
    std::vector<BENCH_DATA_TYPE> data;
    std::vector<BENCH_DATA_TYPE> mean;
    std::vector<BENCH_DATA_TYPE> stddev;
    std::vector<BENCH_DATA_TYPE> symmat;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> data_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mean_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> stddev_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> symmat_buf;

public:
    Correlation(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        data = std::vector<BENCH_DATA_TYPE>((mat_size+1) * (mat_size+1));
        mean = std::vector<BENCH_DATA_TYPE>((mat_size+1));
        stddev = std::vector<BENCH_DATA_TYPE>((mat_size+1));
        symmat = std::vector<BENCH_DATA_TYPE>((mat_size+1)*(mat_size+1));

        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                data[i*mat_size+j] = ((BENCH_DATA_TYPE)i * j) / (mat_size + 1);
            }
        }

        data_buf.initialize(data.data(),     celerity::range<2>((mat_size+1), (mat_size+1)));
        mean_buf.initialize(mean.data(),     celerity::range<2>((mat_size+1), 1));
        stddev_buf.initialize(stddev.data(), celerity::range<2>((mat_size+1), 1));
        symmat_buf.initialize(symmat.data(), celerity::range<2>((mat_size+1), (mat_size+1)));
    }

    void run() {
        correlation(QueueManager::getInstance(),
                    data_buf.get(), mean_buf.get(),
                    stddev_buf.get(), symmat_buf.get(),mat_size);
    }

    static std::string getBenchmarkName() { return "Correlation"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        //todo
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Correlation >();
}
