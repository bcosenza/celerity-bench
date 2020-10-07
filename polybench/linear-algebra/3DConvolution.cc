#include <cstdio>
#include <vector>

#include <common.h>

using BENCH_DATA_TYPE = float;

class Conv3D;

void conv3D(celerity::distr_queue& queue,
            celerity::buffer<BENCH_DATA_TYPE, 3>& mat_a,
            celerity::buffer<BENCH_DATA_TYPE, 3>& mat_b,
            const size_t mat_size) {

    queue.submit([=](celerity::handler& cgh) {
        auto A = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<3>(1,1,1));
        auto B = mat_b.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<3>());
        for(size_t i = 1; i < mat_size - 1; i++){
            cgh.parallel_for<class Conv3D>(cl::sycl::range<3>(1, mat_size-1,mat_size-1),cl::sycl::id<3> {1,1,1}, [=](cl::sycl::item<3> item) {
                const BENCH_DATA_TYPE c11 = +2,  c21 = +5,  c31 = -8;
                const BENCH_DATA_TYPE c12 = -3,  c22 = +6,  c32 = -9;
                const BENCH_DATA_TYPE c13 = +4,  c23 = +7,  c33 = +10;

                const auto j = item[1];
                const auto k = item[2];

                B[item] = c11 * A[{(i - 1), (j - 1), (k - 1)}] + c13 * A[{(i + 1), (j - 1), (k - 1)}] + c21 * A[{(i - 1), (j - 1), (k - 1)}]
                          + c23 * A[{(i + 1), (j - 1), (k - 1)}] + c31 * A[{(i - 1), (j - 1), (k - 1)}] + c33 * A[{(i + 1), (j - 1), (k - 1)}]
                          + c12 * A[{(i + 0), (j - 1), (k + 0)}] + c22 * A[{(i + 0), (j + 0), (k + 0)}] + c32 * A[{(i + 0), (j + 1), (k + 0)}]
                          + c11 * A[{(i - 1), (j - 1), (k + 1)}] + c13 * A[{(i + 1), (j - 1), (k + 1)}] + c21 * A[{(i - 1), (j + 0), (k + 1)}]
                          + c23 * A[{(i + 1), (j + 0), (k + 1)}] + c31 * A[{(i - 1), (j + 1), (k + 1)}] + c33 * A[{(i + 1), (j + 1), (k + 1)}];
            });
        }
    });

}

class Conv3D {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_b;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 3> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 3> mat_b_buf;

public:
    Conv3D(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size * mat_size);
        mat_b = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size * mat_size, 0.f);

        // random seed
        srand(42);
        for(size_t i = 0; i < mat_size; ++i)
            for(size_t j = 0; j < mat_size; ++j)
                for(size_t k = 0; k < mat_size; ++k)
                    mat_a[(i * (mat_size*mat_size)) + (j*mat_size) + k] = (BENCH_DATA_TYPE)rand() / RAND_MAX;

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<3>(mat_size, mat_size,mat_size));
        mat_b_buf.initialize(cl::sycl::range<3>(mat_size, mat_size,mat_size));
    }

    void run() {
        conv3D(QueueManager::getInstance(),
               mat_a_buf.get(), mat_b_buf.get(),
               mat_size);

    }

    static std::string getBenchmarkName() { return "Conv3D"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_b_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<3>(mat_size, mat_size, mat_size));
            cgh.run([=, &verification_passed]() {
                for(size_t i = 1; i < mat_size -1 && verification_passed; ++i)
                    for(size_t j = 1; j < mat_size -1 && verification_passed; ++j)
                        for(size_t k = 1; k < mat_size -1 && verification_passed; ++k){
                            const BENCH_DATA_TYPE c11 = +2,  c21 = +5,  c31 = -8;
                            const BENCH_DATA_TYPE c12 = -3,  c22 = +6,  c32 = -9;
                            const BENCH_DATA_TYPE c13 = +4,  c23 = +7,  c33 = +10;
                            if((i < mat_size - 1) && (j < mat_size - 1) && (i > 0) && (j > 0)){
                                auto res = (c11 * mat_a[((i - 1)*(mat_size*mat_size))+((j - 1)*mat_size)+(k - 1)]
                                            + c13 * mat_a[((i + 1)*(mat_size*mat_size))+((j - 1)*mat_size)+(k - 1)]
                                            + c21 * mat_a[((i - 1)*(mat_size*mat_size))+((j - 1)*mat_size)+(k - 1)]
                                            + c23 * mat_a[((i + 1)*(mat_size*mat_size))+((j - 1)*mat_size)+(k - 1)]
                                            + c31 * mat_a[(i - 1)*(mat_size*mat_size)+((j - 1)*mat_size)+(k - 1)]
                                            + c33 * mat_a[((i + 1)*(mat_size*mat_size))+((j - 1)*mat_size)+(k - 1)]
                                            + c12 * mat_a[((i + 0)*(mat_size*mat_size))+((j - 1)*mat_size)+(k + 0)]
                                            + c22 * mat_a[((i + 0)*(mat_size*mat_size))+((j + 0)*mat_size)+(k + 0)]
                                            + c32 * mat_a[((i + 0)*(mat_size*mat_size))+((j + 1)*mat_size)+(k + 0)]
                                            + c11 * mat_a[((i - 1)*(mat_size*mat_size))+((j - 1)*mat_size)+(k + 1)]
                                            + c13 * mat_a[((i + 1)*(mat_size*mat_size))+((j - 1)*mat_size)+(k + 1)]
                                            + c21 * mat_a[((i - 1)*(mat_size*mat_size))+((j + 0)*mat_size)+(k + 1)]
                                            + c23 * mat_a[((i + 1)*(mat_size*mat_size))+((j + 0)*mat_size)+(k + 1)]
                                            + c31 * mat_a[((i - 1)*(mat_size*mat_size))+((j + 1)*mat_size)+(k + 1)]
                                            + c33 * mat_a[((i + 1)*(mat_size*mat_size))+((j + 1)*mat_size)+(k + 1)]);
                                verification_passed = almost_equal(result[i][j][k], res, .1f);
                                if(!verification_passed)
                                    std::cout<<result[i][j][k]<<"\t"<<res<<std::endl;
                            }
                        }

            });
        });
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Conv3D >();
}
