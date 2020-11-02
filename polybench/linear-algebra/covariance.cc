#include <cstdio>
#include <vector>

#include <common.h>
#define FLOAT_N 3214212.01
    
#define sqrt_of_array_cell(x, j) sqrt(x[j])

using BENCH_DATA_TYPE = float;

class Correlation;

void correlation(celerity::distr_queue& queue,
                 celerity::buffer<BENCH_DATA_TYPE, 2>& d,
                 celerity::buffer<BENCH_DATA_TYPE, 2>& m,
                 celerity::buffer<BENCH_DATA_TYPE, 2>& sd,
                 const size_t mat_size) {

    using namespace cl::sycl;
    using namespace celerity::access;

    queue.submit([=](celerity::handler& cgh) {
        auto data = d.template get_access<access::mode::read>(cgh, slice<2>(0));
        auto mean = m.template get_access<access::mode::discard_write>(cgh, one_to_one<2>());
        cgh.parallel_for<class Covariance1>(range<2>(mat_size, 1), id<2>(1, 0), [=, N_ = mat_size](item<2> item) {
            const auto j = item[0];

            BENCH_DATA_TYPE result = 0;
            for(size_t i = 1; i <= N_; i++) {
                result += data[{i, j}];
            }
            mean[item] = result / FLOAT_N;
        });
    });

    queue.submit([=](celerity::handler& cgh) {
        auto mean = m.template get_access<access::mode::read>(cgh, slice<2>(1));
        auto data = d.template get_access<access::mode::read_write>(cgh, one_to_one<2>());
        cgh.parallel_for<class Covariance2>(range<2>(mat_size, mat_size), id<2>(1, 1), [=](item<2> item) {
            const auto j = item[1];
            data[item] -= mean[{j, 0}];
        });
    });

    queue.submit([=](celerity::handler& cgh) {
        auto data = d.template get_access<access::mode::read>(cgh, celerity::access::all<2>());
        auto symmat = sd.template get_access<access::mode::discard_write>(cgh, slice<2>(1));
        auto symmat2 = sd.template get_access<access::mode::discard_write>(cgh, slice<2>(0));
        cgh.parallel_for<class Covariance3>(range<2>(mat_size, 1), id<2>(1, 0), [=, M_ = mat_size, N_ = mat_size](item<2> item) {
            const auto j1 = item[0];

            symmat[{j1, j1}] = 1.0;

            for(size_t j2 = j1; j2 <= M_; j2++) {
                BENCH_DATA_TYPE result = 0.0;
                for(size_t i = 1; i <= N_; i++) {
                    result += data[{i, j1}] * data[{i, j2}];
                }

                symmat[{j1, j2}] = result;
                symmat2[{j2, j1}] = result;
            }
        });
    });
}


class Correlation {
protected:
    std::vector<BENCH_DATA_TYPE> data;
    std::vector<BENCH_DATA_TYPE> mean;
    std::vector<BENCH_DATA_TYPE> symmat;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> data_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mean_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> symmat_buf;

public:
    Correlation(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        data = std::vector<BENCH_DATA_TYPE>((mat_size+1) * (mat_size+1));
        mean = std::vector<BENCH_DATA_TYPE>((mat_size+1));
        symmat = std::vector<BENCH_DATA_TYPE>((mat_size+1)*(mat_size+1));

        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                data[i*(mat_size+1)+j] = ((BENCH_DATA_TYPE)i * j) / (mat_size);
            }
        }

        data_buf.initialize(data.data(), cl::sycl::range<2>((mat_size+1), (mat_size+1)));
        mean_buf.initialize(mean.data(), cl::sycl::range<2>((mat_size+1), 1));
        symmat_buf.initialize(symmat.data(), cl::sycl::range<2>((mat_size+1), (mat_size+1)));
    }

    void run() {
        correlation(QueueManager::getInstance(),
                    data_buf.get(), mean_buf.get()
                    , symmat_buf.get(),mat_size);
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
