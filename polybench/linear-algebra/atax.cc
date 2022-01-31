#include <cstdio>
#include <vector>

#include <common.h>

using BENCH_DATA_TYPE = float;

class Atax;

void atax(celerity::distr_queue queue, celerity::buffer<BENCH_DATA_TYPE, 2> mat_a, \
                celerity::buffer<BENCH_DATA_TYPE, 2> mat_x, celerity::buffer<BENCH_DATA_TYPE, 2> mat_y, 
                celerity::buffer<BENCH_DATA_TYPE, 2> mat_tmp,const size_t mat_size) {

#if BENCH_KERNEL == 1 || !defined( BENCH_KERNEL )
    queue.submit([=](celerity::handler& cgh) {
        celerity::accessor y{mat_y, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
        cgh.parallel_for<class Atax1>(celerity::range<2>(mat_size, 1), [=](celerity::item<2> item) { y[item] = 0; });
    });
#endif
#if BENCH_KERNEL == 2 || !defined( BENCH_KERNEL )
    queue.submit([=](celerity::handler& cgh) {
        celerity::accessor A{mat_a, cgh, celerity::access::slice<2>(1), celerity::read_only};
        celerity::accessor x{mat_x, cgh, celerity::access::all{}, celerity::read_only};
        celerity::accessor tmp{mat_tmp, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
        cgh.parallel_for<class Atax2>(celerity::range<2>(mat_size, 1), [=, NY_ = mat_size ](celerity::item<2> item) {
            const auto i = item[0];

            BENCH_DATA_TYPE result = 0;
            for(size_t j = 0; j < NY_; j++) {
                result += A[{i, j}] * x[{j, 0}];
            }
            tmp[item] = result;
        });
    });
#endif
#if BENCH_KERNEL == 3 || !defined( BENCH_KERNEL )
    queue.submit([=](celerity::handler& cgh) {
        celerity::accessor A{mat_a, cgh, celerity::access::slice<2>(0), celerity::read_only};
        celerity::accessor tmp{mat_tmp, cgh, celerity::access::all{}, celerity::read_only};
        celerity::accessor y{mat_y, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};

        cgh.parallel_for<class Atax3>(celerity::range<2>(mat_size, 1), [=, NX_ = mat_size](celerity::item<2> item) {
            const auto j = item[0];

            BENCH_DATA_TYPE result = 0;
            for(size_t i = 0; i < NX_; i++) {
                result += A[{i, j}] * tmp[{i, 0}];
            }
            y[item] = result;
        });
    });
#endif
}

class Atax {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_x;
    std::vector<BENCH_DATA_TYPE> mat_y;
    std::vector<BENCH_DATA_TYPE> mat_tmp;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_x_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_y_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_tmp_buf;

public:
    Atax(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_x = std::vector<BENCH_DATA_TYPE>(mat_size);
        mat_y = std::vector<BENCH_DATA_TYPE>(mat_size);
        mat_tmp = std::vector<BENCH_DATA_TYPE>(mat_size);

        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                mat_a[i * mat_size + j] = ((BENCH_DATA_TYPE)i * (j)) / mat_size;
            }
        }

        mat_a_buf.initialize(mat_a.data(), celerity::range<2>(mat_size, mat_size));
        mat_x_buf.initialize(mat_x.data(), celerity::range<2>(mat_size, 1));
        mat_y_buf.initialize(mat_y.data(), celerity::range<2>(mat_size, 1));
        mat_tmp_buf.initialize(            celerity::range<2>(mat_size, 1));
    }

    void run() {
        atax(QueueManager::getInstance(), mat_a_buf.get(), mat_x_buf.get(),mat_y_buf.get(),mat_tmp_buf.get(),mat_size);
    }

    static std::string getBenchmarkName() { return "Atax"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        /*QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_y_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size,1));
            cgh.run([=, &verification_passed]() {
                std::vector<BENCH_DATA_TYPE> test_y(mat_size,0.f);
                std::vector<BENCH_DATA_TYPE> test_tmp(mat_size,0.f);
                for(size_t i = 0; i < mat_size && verification_passed; ++i){
                    for(size_t j = 0; j < mat_size; ++j)
                        test_tmp[i] = test_tmp[i] + mat_a[i * mat_size + j] * mat_x[j];
                    for(size_t j = 0; j < mat_size; ++j)
                        test_y[j] = test_y[j] + mat_a[i * mat_size + j] * test_tmp[i];

                    verification_passed =  almost_equal(test_y[i], result[{i,0}],3);
                }
            });
        });*/
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);
    app.run< Atax >();
}
