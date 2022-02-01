#include <cstdio>
#include <vector>

#include <common.h>

using BENCH_DATA_TYPE = float;

class Gramschmidt;

void gramschmidt(celerity::distr_queue queue, celerity::buffer<BENCH_DATA_TYPE, 2> mat_a,
              celerity::buffer<BENCH_DATA_TYPE, 2> mat_r,
              celerity::buffer<BENCH_DATA_TYPE, 2> mat_q, const size_t mat_size) {

    using namespace cl::sycl;
    using namespace celerity::access;

    for(size_t k = 0; k < mat_size; k++) {
        const auto non_empty_chunk_range_mapper = [=](celerity::chunk<2> chunk) -> celerity::subrange<2> {
            if (chunk.range[0] == 0) return {};
            return {{k, k},
                    {1, 1}};
        };

#if BENCH_KERNEL == 1 || !defined( BENCH_KERNEL )
        queue.submit([=](celerity::handler &cgh) {
            celerity::accessor A{mat_a, cgh, celerity::access::slice<2>(0), celerity::read_only};
            //celerity::accessor R{mat_r, cgh, non_empty_chunk_range_mapper, celerity::write_only, celerity::no_init};
            celerity::accessor R{mat_r, cgh, celerity::access::fixed<2>({{k, k}, {1, 1}}), celerity::write_only, celerity::no_init};

            cgh.parallel_for<class Gramschmidt1>(range<2>(1, 1), [=, M_ = mat_size](celerity::item<2> item) {
                BENCH_DATA_TYPE nrm = 0;
                for (size_t i = 0; i < M_; i++) {
                    nrm += A[{i, k}] * A[{i, k}];
                }
                R[{k, k}] = sqrt(nrm);
            });
        });
#endif
#if BENCH_KERNEL == 2 || !defined( BENCH_KERNEL )
        queue.submit([=](celerity::handler &cgh) {
            celerity::accessor A{mat_a, cgh, celerity::access::one_to_one{}, celerity::read_only};
            celerity::accessor R{mat_r, cgh, celerity::access::fixed<2>({{k, k}, {1, 1}}), celerity::read_only};
            celerity::accessor Q{mat_q, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
            cgh.parallel_for<class Gramschmidt2>(range<2>(mat_size, 1), id<2>(0, k), [=](celerity::item<2> item) {
                Q[item] = A[item] / R[{k, k}];
            });
        });
#endif
#if BENCH_KERNEL == 3 || !defined( BENCH_KERNEL )
        queue.submit([=](celerity::handler &cgh) {
            celerity::accessor A{mat_a, cgh, celerity::access::slice<2>(0), celerity::read_write};
            celerity::accessor R{mat_r, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
            celerity::accessor Q{mat_q, cgh, celerity::access::slice<2>(0), celerity::read_only};
            cgh.parallel_for<class Gramschmidt3>(range<2>(1, mat_size - k - 1), id<2>(k, k + 1), [=, M_ = mat_size](celerity::item<2> item) {
                const auto k = item[0];
                const auto j = item[1];

                BENCH_DATA_TYPE R_result = 0;
                for (size_t i = 0; i < M_; i++) {
                    R_result += Q[{i, k}] * A[{i, j}];
                }
                R_result;

                for (size_t i = 0; i < M_; i++) {
                    A[{i, j}] -= Q[{i, k}] * R_result;
                }

                R[{k, j}] = R_result;
            });
        });
#endif
    }
}


class Gramschmidt {
protected:
    std::vector<BENCH_DATA_TYPE> mat_a;
    std::vector<BENCH_DATA_TYPE> mat_r;
    std::vector<BENCH_DATA_TYPE> mat_q;
    BenchmarkArgs args;
    int mat_size;

    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_r_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_q_buf;

public:
    Gramschmidt(const BenchmarkArgs &_args) : args(_args) {
        mat_size = args.problem_size;
    }

    void setup() {
        mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_r = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
        mat_q = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);

        for(size_t i = 0; i < mat_size; ++i) {
            for(size_t j = 0; j < mat_size; ++j) {
                mat_a[i * mat_size + j] = ((BENCH_DATA_TYPE)(i + 1) * (j + 1)) / (mat_size + 1);
            }
        }

        mat_a_buf.initialize(mat_a.data(), celerity::range<2>(mat_size, mat_size));
        mat_r_buf.initialize(celerity::range<2>(mat_size, mat_size));
        mat_q_buf.initialize(celerity::range<2>(mat_size, mat_size));
    }

    void run() {
        gramschmidt(QueueManager::getInstance(),
                    mat_a_buf.get(),mat_r_buf.get(),
                    mat_q_buf.get(), mat_size);
    }

    static std::string getBenchmarkName() { return "Gramschmidt"; }

    bool verify(VerificationSetting &ver) {
        bool verification_passed = true;
        //TODO
        QueueManager::sync();
        return verification_passed;
    }
};

int main(int argc, char** argv) {
    BenchmarkApp app(argc, argv);

    app.run< Gramschmidt >();
}
