#include <vector>

#include <common.h>
class Fdtd_apml;

using BENCH_DATA_TYPE = float;

void fdtd_apml(celerity::distr_queue& queue,
               celerity::buffer<BENCH_DATA_TYPE, 1>& mat_a, celerity::buffer<BENCH_DATA_TYPE, 1>& mat_res,
               const size_t mat_size){
    queue.submit([=](celerity::handler& cgh) {
        auto A = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<1>(1));
        auto RES = mat_res.template get_access<cl::sycl::access::mode::write>(cgh, celerity::access::neighborhood<1>(1));
        cgh.parallel_for<class Fdtd_apml>(cl::sycl::range<1> (mat_size - 1), cl::sycl::id<1> {1}, [=](cl::sycl::item<1> item) {
            auto i = item[0];
            RES[i] =  0.33333 * (A[i-1] + A[i] + A[i + 1]);
        });
    });
}

class Fdtd_apml {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_res;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 1> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 1> mat_res_buf;

public:
    Fdtd_apml(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size);
        mat_res = std::vector<BENCH_DATA_TYPE>(mat_size);

        for (size_t i = 0; i < mat_size; i++){
            mat_a[i] = ((BENCH_DATA_TYPE) i+ 2) / mat_size;
            mat_res[i] = ((BENCH_DATA_TYPE) i+ 3) / mat_size;
        }

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<1>(mat_size));
        mat_res_buf.initialize(mat_res.data(), cl::sycl::range<1>(mat_size));
    }

    void run() {
        fdtd_apml(QueueManager::getInstance(), mat_a_buf.get(), mat_res_buf.get(), mat_size);
    }

    static std::string getBenchmarkName() { return "Fdtd_apml"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = mat_res_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(mat_size));
            cgh.run([=, &verification_passed]() {
                for(size_t i = 1; i < mat_size -1 && verification_passed; ++i){
                    const float kernel_value = result[i];
                    const float host_value =  0.33333 * (mat_a[i-1] + mat_a[i] + mat_a[i + 1]);
                    verification_passed = almost_equal(kernel_value,host_value, 0.05f);
                    if(!verification_passed)
                        std::cout<<std::setprecision(20)<<host_value<<"\t"<<kernel_value<<std::endl;
                }
            });
        });
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);
    app.run< Fdtd_apml >();
}
