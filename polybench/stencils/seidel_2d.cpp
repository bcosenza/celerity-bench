#include <vector>

#include <common.h>
class Seidel;

using BENCH_DATA_TYPE = float;

void seidel(celerity::distr_queue& queue,
            celerity::buffer<BENCH_DATA_TYPE, 2>& mat_a,
            const size_t mat_size){
    queue.submit([=](celerity::handler& cgh) {
        auto A = mat_a.template get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::neighborhood<2>(1,1));
        cgh.parallel_for<class Seidel>(cl::sycl::range<2> (mat_size - 2, mat_size - 2), cl::sycl::id<2> {1,1}, [=](cl::sycl::item<2> item) {
            auto i = item[0];
            auto j = item[1];
            A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                       + A[i][j-1] + A[i][j] + A[i][j+1]
                       + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/9.0;
        });
    });
}

class Seidel {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;

public:
    Seidel(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);

        for (size_t i = 0; i < mat_size; i++)
            for (size_t j = 0; j < mat_size; j++)
                mat_a[(i*mat_size)+j] = ((BENCH_DATA_TYPE) i*(j+2) + 2) / mat_size;

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));

    }

    void run() {
        seidel(QueueManager::getInstance(), mat_a_buf.get(),mat_size);
    }

    static std::string getBenchmarkName() { return "Seidel"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        auto test = std::vector<std::vector<BENCH_DATA_TYPE>>(mat_size,std::vector<BENCH_DATA_TYPE>(mat_size));
        for (size_t i = 0; i < mat_size; i++)
            for (size_t j = 0; j < mat_size; j++)
                test[i][j] = ((BENCH_DATA_TYPE) i*(j+2) + 2) / mat_size;
        QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_a_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));
            cgh.run([=, &verification_passed]() {
                for(size_t i = 1; i < mat_size -2 && verification_passed; ++i)
                    for(size_t j = 1; j < mat_size -2 && verification_passed; ++j) {
                        const float kernel_value = result[{i, j}];
                        const float host_value = (test[i-1][j-1] + test[i-1][j] + test[i-1][j+1]
                                                  + test[i][j-1] + test[i][j] + test[i][j+1]
                                                  + test[i+1][j-1] + test[i+1][j] + test[i+1][j+1])/9.0;
                        verification_passed = almost_equal(kernel_value,host_value,5);

                    }
            });
        });
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);
    app.run< Seidel >();
}
