#include <cstdio>
#include <vector>

#include <common.h>

using BENCH_DATA_TYPE = float;

class Mvt;

void mvt(celerity::distr_queue& queue,
         celerity::buffer<BENCH_DATA_TYPE, 2>& mat_a,
         celerity::buffer<BENCH_DATA_TYPE, 2>& mat_x1,
         celerity::buffer<BENCH_DATA_TYPE, 2>& mat_x2,
         celerity::buffer<BENCH_DATA_TYPE, 2>& mat_y1,
         celerity::buffer<BENCH_DATA_TYPE, 2>& mat_y2,
         const size_t mat_size) {

    using namespace cl::sycl;
    using namespace celerity::access;
    queue.submit([=](celerity::handler& cgh) {
        auto a = mat_a.template get_access<access::mode::read>(cgh, slice<2>(1));
        auto y1 = mat_y1.template get_access<access::mode::read>(cgh, slice<2>(0));
        auto x1 = mat_y2.template get_access<access::mode::read_write>(cgh, one_to_one<2>());
        cgh.parallel_for<class Mvt1>(mat_x1.get_range(), [=, N_ = mat_size](item<2> item) {
            const auto i = item[0];
            for(size_t j = 0; j < N_; j++) {
                x1[{i, 0}] += a[{i, j}] * y1[{j, 0}];
            }
        });
    });

    queue.submit([=](celerity::handler& cgh) {
        auto a = mat_a.template get_access<access::mode::read>(cgh, slice<2>(1));
        auto y2 = mat_y2.template get_access<access::mode::read>(cgh, slice<2>(0));
        auto x2 = mat_x2.template get_access<access::mode::read_write>(cgh, one_to_one<2>());
        cgh.parallel_for<class Mvt2>(mat_x1.get_range(), [=, N_ = mat_size](item<2> item) {
            const auto k = item[0];
            for(size_t l = 0; l < N_; l++) {
                x2[{k, 0}] += a[{k, l}] * y2[{l, 0}];
            }
        });
    });


}


class Mvt {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_x1;
    std::vector<BENCH_DATA_TYPE> mat_x2;
    std::vector<BENCH_DATA_TYPE> mat_y1;
    std::vector<BENCH_DATA_TYPE> mat_y2;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_x1_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_x2_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_y1_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_y2_buf;

public:
    Mvt(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_x1 = std::vector<BENCH_DATA_TYPE>(mat_size, 0);
        mat_x2 = std::vector<BENCH_DATA_TYPE>(mat_size, 0);
        mat_y1 = std::vector<BENCH_DATA_TYPE>(mat_size, 0);
        mat_y2 = std::vector<BENCH_DATA_TYPE>(mat_size, 0);

        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                mat_a[i * mat_size + j] = (BENCH_DATA_TYPE)(i + j + 1.0) / mat_size;
            }
        }

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_x1_buf.initialize(mat_x1.data(), cl::sycl::range<2>(mat_size, 1));
        mat_x2_buf.initialize(mat_x2.data(), cl::sycl::range<2>(mat_size, 1));
        mat_y1_buf.initialize(mat_y1.data(), cl::sycl::range<2>(mat_size, 1));
        mat_y2_buf.initialize(mat_y2.data(), cl::sycl::range<2>(mat_size, 1));
    }

    void run() {
        mvt(QueueManager::getInstance(),
            mat_a_buf.get(),mat_x1_buf.get(),
                mat_x2_buf.get(),mat_y1_buf.get(),
                mat_y2_buf.get(),mat_size);
    }

    static std::string getBenchmarkName() { return "Mvt"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result_x1 = mat_x1_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, 1));
            auto result_x2 = mat_x2_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, 1));
            cgh.run([=, &verification_passed]() {
                std::vector<BENCH_DATA_TYPE> res_x1(mat_size, 0);
                std::vector<BENCH_DATA_TYPE> res_x2(mat_size, 0);
                for(size_t i = 0; i < mat_size && verification_passed; ++i)
                    for(size_t j = 0; j < mat_size && verification_passed; ++j){
                        for(size_t k = 0; k < mat_size; k++) {
                            res_x1[i] += mat_a[(i*mat_size) + k] * mat_y1[k];
                        }
                        for(size_t k = 0; k < mat_size; k++) {
                            res_x2[i] += mat_a[i*mat_size + k] * mat_y2[k];
                        }
                        verification_passed = res_x1[i] == result_x1[i][0];
                        verification_passed = res_x2[i] == result_x2[i][0];
                    }
            });
        });
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Mvt >();
}
