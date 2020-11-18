#include <vector>

#include <common.h>
class Adi;

using BENCH_DATA_TYPE = float;

class Adi {
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
    Adi(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_b = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_res = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);

        for (size_t i = 0; i < mat_size; i++)
            for (size_t j = 0; j < mat_size; j++){
                mat_a[(i*mat_size)+j]=((BENCH_DATA_TYPE) i*(j+1) + 1) / mat_size;
                mat_b[(i*mat_size)+j]=((BENCH_DATA_TYPE) i*(j+2) + 2) / mat_size;
                mat_res[(i*mat_size)+j]=((BENCH_DATA_TYPE) i*(j+3) + 3) / mat_size;
            }

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_b_buf.initialize(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_res_buf.initialize(mat_res.data(), cl::sycl::range<2>(mat_size, mat_size));
    }

    void run() {
        celerity::distr_queue& queue = QueueManager::getInstance();
        queue.submit([=](celerity::handler& cgh) {
            auto A = mat_a_buf.get().template get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::neighborhood<2>(1,1));
            auto B = mat_b_buf.get().template get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::neighborhood<2>(1,1));
            auto RES = mat_res_buf.get().template get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::neighborhood<2>(1,1));

            cgh.parallel_for<class Adi>(cl::sycl::range<2> (mat_size, mat_size), cl::sycl::id<2> {0,1}, [=](cl::sycl::item<2> item) {
                auto i = item[0];
                auto j = item[1];
                RES[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
            });

            cgh.parallel_for<class Adi>(cl::sycl::range<2> (mat_size - 1, mat_size - 1), cl::sycl::id<2> {0,1}, [=](cl::sycl::item<2> item) {
                auto i = item[0];
                auto j = item[1];
                RES[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
            });

            cgh.parallel_for<class Adi>(cl::sycl::range<2> (mat_size - 1, mat_size - 1), cl::sycl::id<2> {0,1}, [=](cl::sycl::item<2> item) {
                auto i = item[0];
                auto j = item[1];
                RES[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
            });

            cgh.parallel_for<class Adi>(cl::sycl::range<2> (mat_size - 1, mat_size - 1), cl::sycl::id<2> {0,1}, [=](cl::sycl::item<2> item) {
                auto i = item[0];
                auto j = item[1];
                RES[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
            });

            cgh.parallel_for<class Adi>(cl::sycl::range<2> (mat_size - 1, mat_size - 1), cl::sycl::id<2> {0,1}, [=](cl::sycl::item<2> item) {
                auto i = item[0];
                auto j = item[1];
                RES[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
            });
        });
        QueueManager::sync();
    }

    static std::string getBenchmarkName() { return "Adi"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_res_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));
            cgh.run([=, &verification_passed]() {
                for(size_t i = 1; i < mat_size -1 && verification_passed; ++i)
                    for(size_t j = 1; j < mat_size -1 && verification_passed; ++j) {
                        const float kernel_value = result[{i, j}];
                        const float host_value = 0.2 * (mat_a[(i*mat_size)+j] + mat_a[(i*mat_size)+(j-1)]
                                                        + mat_a[(i*mat_size)+(1+j)] + mat_a[((i+1)*mat_size)+j]
                                                        + mat_a[((i-1)*mat_size)+j]);
                        verification_passed = kernel_value != host_value;
                    }
            });
        });
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);
    app.run< Adi >();
}
