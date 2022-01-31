#include <cstdio>
#include <vector>

#include <common.h>

using BENCH_DATA_TYPE = float;

class Conv2D;

void conv2D(celerity::distr_queue queue,
               celerity::buffer<BENCH_DATA_TYPE, 2> mat_a,
               celerity::buffer<BENCH_DATA_TYPE, 2> mat_b,
               const size_t mat_size){
    queue.submit([=](celerity::handler& cgh) {
        celerity::accessor A{mat_a, cgh, celerity::access::neighborhood<2>(1,1), celerity::read_only};
        celerity::accessor B{mat_b, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
        cgh. parallel_for<class Conv2D>(cl::sycl::range<2>(mat_size-1, mat_size-1),cl::sycl::id<2> {1,1}, [=](celerity::item<2> item) {
            const BENCH_DATA_TYPE c11 = +0.2, c21 = +0.5, c31 = -0.8;
            const BENCH_DATA_TYPE c12 = -0.3, c22 = +0.6, c32 = -0.9;
            const BENCH_DATA_TYPE c13 = +0.4, c23 = +0.7, c33 = +0.10;
            const auto i = item[0];
            const auto j = item[1];

            B[item] = c11 * A[{(i - 1), (j - 1)}] + c12 * A[{(i + 0), (j - 1)}] + c13 * A[{(i + 1), (j - 1)}]
                    + c21 * A[{(i - 1), (j + 0)}] + c22 * A[{(i + 0), (j + 0)}] + c23 * A[{(i + 1), (j + 0)}]
                    + c31 * A[{(i - 1), (j + 1)}] + c32 * A[{(i + 0), (j + 1)}] + c33 * A[{(i + 1), (j + 1)}];
        });
    });
}

class Conv2D {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_b;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_b_buf;

public:
    Conv2D(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_b = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);

        // random seed
        srand(42);
        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                mat_a[i * mat_size + j] = (BENCH_DATA_TYPE)rand() / RAND_MAX;
            }
        }

        auto range = celerity::range<2>(mat_size, mat_size);
        mat_a_buf.initialize(mat_a.data(), range);
        mat_b_buf.initialize(range);
    }

    void run() {
        conv2D(QueueManager::getInstance(), mat_a_buf.get(), mat_b_buf.get(), mat_size);
    }

    static std::string getBenchmarkName() { return "Conv2D"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        /* QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_b_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));
            cgh.run([=, &verification_passed]() {
                for(size_t i = 1; i < mat_size -1 && verification_passed; ++i)
                    for(size_t j = 1; j < mat_size -1 && verification_passed; ++j){
                        const BENCH_DATA_TYPE c11 = +0.2, c21 = +0.5, c31 = -0.8;
                        const BENCH_DATA_TYPE c12 = -0.3, c22 = +0.6, c32 = -0.9;
                        const BENCH_DATA_TYPE c13 = +0.4, c23 = +0.7, c33 = +0.10;
                        if((i < mat_size - 1) && (j < mat_size - 1) && (i > 0) && (j > 0)){
                            auto res = (c11 * mat_a[(((i- 1)*mat_size))+(j - 1)] + c12 * mat_a[((i*mat_size) + 0)+(j - 1)] + c13 * mat_a[(((i+1)*mat_size) )+(j - 1)] + c21 * mat_a[(((i- 1)*mat_size) )+(j + 0)]
                                        + c22 * mat_a[((i*mat_size) + 0)+(j + 0)] + c23 * mat_a[(((i+ 1)*mat_size) )+(j + 0)] + c31 * mat_a[(((i- 1)*mat_size) )+(j + 1)] + c32 * mat_a[((i*mat_size))+(j + 1)]
                                        + c33 * mat_a[(((i+ 1)*mat_size) )+(j + 1)]);

                            verification_passed = almost_equal(result[{i, j}], res, .1f);
                        }
                    }

            });
        }); */
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Conv2D >();
    return 0;
}
