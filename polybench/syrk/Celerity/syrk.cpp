#include <vector>

#include <cstdlib>

#include <celerity/celerity.h>

#include "polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

// Problem size
auto N = 1024;
auto M = 1024;

using DATA_TYPE = float;

namespace values {
constexpr DATA_TYPE alpha = 123;
constexpr DATA_TYPE beta = 14512;
} // namespace values

void compareResults(const DATA_TYPE* C, const DATA_TYPE* C_outputFromGpu) {
	int i, j, fail;
	fail = 0;

	for(i = 0; i < N; i++) {
		for(j = 0; j < M; j++) {
			if(percentDiff(C[i * M + j], C_outputFromGpu[i * M + j]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_arrays(DATA_TYPE* A, DATA_TYPE* C) {
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < M; j++) {
			A[i * M + j] = ((DATA_TYPE)i * j) / N;
		}

		for(j = 0; j < N; j++) {
			C[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
		}
	}
}

void syrk(DATA_TYPE* A, DATA_TYPE* C) {
	int i, j, k;

	/*  C := alpha*A*A' + beta*C */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			C[i * M + j] *= values::beta;
		}
	}

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < M; k++) {
				C[i * N + j] += values::alpha * A[i * M + k] * A[j * M + k];
			}
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		N = problem_size;
		M = problem_size;
	}
	std::cout << "Problem size: " << N << "\n";

	std::vector<DATA_TYPE> A(N * M);
	std::vector<DATA_TYPE> C(N * M);

	init_arrays(A.data(), C.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		syrk(A.data(), C.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;
		using namespace celerity::access;

		std::vector<DATA_TYPE> C_gpu(N * M);

		init_arrays(A.data(), C_gpu.data());

		celerity::buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(N, M));
		celerity::buffer<DATA_TYPE, 2> C_buffer(C_gpu.data(), range<2>(N, M));

		celerity::distr_queue queue;
		queue.slow_full_sync();
		double t_start = rtclock();

		queue.submit([=](celerity::handler& cgh) {
			auto C = C_buffer.get_access<access::mode::read_write>(cgh, one_to_one<2>());
			cgh.parallel_for<class Syrk1>(C_buffer.get_range(), [=](item<2> item) { C[item] *= values::beta; });
		});

		queue.submit([=](celerity::handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh, slice<2>(1));
			auto C = C_buffer.get_access<access::mode::read_write>(cgh, one_to_one<2>());
			cgh.parallel_for<class Syr2k2>(C_buffer.get_range(), [=, M_ = M](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				for(size_t k = 0; k < M_; k++) {
					C[item] += values::alpha * A[{i, k}] * A[{j, k}];
				}
			});
		});

		queue.slow_full_sync();
		double t_end = rtclock();

		queue.with_master_access([=, &C](celerity::handler& cgh) {
			auto C_gpu_result = C_buffer.get_access<access::mode::read>(cgh, C_buffer.get_range());
			cgh.run([=, &C]() {
				fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
				if(shouldDoCpu()) compareResults(C.data(), C_gpu_result.get_pointer());
			});
		});
	}
}
