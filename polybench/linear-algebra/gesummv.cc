#include <cstdio>
#include <vector>

#include <common.h>
using BENCH_DATA_TYPE = float;

namespace values {
    constexpr BENCH_DATA_TYPE alpha = 32412;
    constexpr BENCH_DATA_TYPE beta = 2123;
} // namespace values

void gesummv(celerity::distr_queue queue,
             celerity::buffer<BENCH_DATA_TYPE, 2> mat_a,
             celerity::buffer<BENCH_DATA_TYPE, 2> mat_b,
             celerity::buffer<BENCH_DATA_TYPE, 1> vec_x,
             celerity::buffer<BENCH_DATA_TYPE, 1> vec_y,
             celerity::buffer<BENCH_DATA_TYPE, 1> vec_tmp,
             const size_t mat_size){

    queue.submit([=](celerity::handler &cgh) {
        const auto rows = [=](celerity::chunk<1> chunk) -> celerity::subrange<2> {
            return {{chunk.offset[0], 0}, {chunk.range[0], (size_t)mat_size}};
        };

        celerity::accessor a{mat_a, cgh, rows, celerity::read_only};
        celerity::accessor b{mat_b, cgh, rows, celerity::read_only};
        celerity::accessor x{vec_x, cgh, celerity::access::one_to_one{}, celerity::read_only};
        celerity::accessor y{vec_y, cgh, celerity::access::one_to_one{}, celerity::read_write};
        celerity::accessor tmp{vec_tmp, cgh, celerity::access::one_to_one{}, celerity::read_write};
        cgh.parallel_for<class Gesummv>(cl::sycl::range<1>(mat_size), [=, n = mat_size](celerity::item<1> item) {
            const auto i = item[0];
            BENCH_DATA_TYPE tmp_result = tmp[item];
            BENCH_DATA_TYPE y_result = y[item];

            for(size_t j = 0; j < n; j++) {
                tmp_result += a[{i, j}] * x[j];
                y_result += b[{i, j}] * x[j];
            }

            tmp[item] = tmp_result;
            y[item] = values::alpha * tmp_result + values::beta * y_result;
        });
    });
}

class Gesummv;

class Gesummv {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_b;
    std::vector<BENCH_DATA_TYPE> x;
    std::vector<BENCH_DATA_TYPE> y;
    std::vector<BENCH_DATA_TYPE> tmp;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_b_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 1> x_buffer;
    PrefetchedBuffer<BENCH_DATA_TYPE, 1> y_buffer;
    PrefetchedBuffer<BENCH_DATA_TYPE, 1> tmp_buf;

public:
    Gesummv(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_b = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        x = std::vector<BENCH_DATA_TYPE>(mat_size);
        y = std::vector<BENCH_DATA_TYPE>(mat_size,0.f);
        tmp = std::vector<BENCH_DATA_TYPE>(mat_size, 0.f);

        for(size_t i = 0; i < mat_size; i++) {
            x[i] = ((BENCH_DATA_TYPE)i) / mat_size;

            for(size_t j = 0; j < mat_size; j++) {
                mat_a[i * mat_size + j] = ((BENCH_DATA_TYPE)i * j) / mat_size;
                mat_b[i * mat_size + j] = ((BENCH_DATA_TYPE)i * j) / mat_size;
            }
        }

        mat_a_buf.initialize(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
        mat_b_buf.initialize(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
        x_buffer.initialize(x.data(), cl::sycl::range<1>(mat_size));
        y_buffer.initialize(y.data(), cl::sycl::range<1>(mat_size));
        tmp_buf.initialize(cl::sycl::range<1>(mat_size));
    }

    void run() {
        gesummv(QueueManager::getInstance(),
                mat_a_buf.get(), mat_b_buf.get(),
                x_buffer.get(), y_buffer.get(),
                tmp_buf.get(), mat_size);
    }

    static std::string getBenchmarkName() { return "Gesummv"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        /*QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
            auto result = y_buffer.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(mat_size));

            for(size_t i = 0; i < mat_size; i++) {
                tmp[i] = 0;
                y[i] = 0;
                for(size_t j = 0; j < mat_size; j++) {
                    tmp[i] = mat_a[i * mat_size + j] * x[j] + tmp[i];
                    y[i] = mat_b[i * mat_size + j] * x[j] + y[i];
                }

                y[i] = values::alpha * tmp[i] + values::beta * y[i];

                verification_passed = almost_equal(y[i], result[i],3);

            }
        });*/
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Gesummv >();
}
