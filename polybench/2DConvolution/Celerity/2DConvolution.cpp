#include <vector>

#include <cstdlib>

#include <celerity/celerity.h>

#include "polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

// Problem size
auto NI = 4096;
auto NJ = 4096;

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* B, const DATA_TYPE* B_outputFromGpu) {
	int i, j, fail;
	fail = 0;

	for(i = 1; i < (NI - 1); i++) {
		for(j = 1; j < (NJ - 1); j++) {
			if(percentDiff(B[i * NJ + j], B_outputFromGpu[i * NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init(DATA_TYPE* A) {
	int i, j;

	for(i = 0; i < NI; ++i) {
		for(j = 0; j < NJ; ++j) {
			A[i * NJ + j] = (float)rand() / RAND_MAX;
		}
	}
}

void conv2D(DATA_TYPE* A, DATA_TYPE* B) {
	int i, j;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	// clang-format off
	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;
	// clang-format on


	for(i = 1; i < NI - 1; ++i) // 0
	{
		for(j = 1; j < NJ - 1; ++j) // 1
		{
			B[i * NJ + j] = c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] + c13 * A[(i + 1) * NJ + (j - 1)]
			                + c21 * A[(i - 1) * NJ + (j + 0)] + c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)]
			                + c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] + c33 * A[(i + 1) * NJ + (j + 1)];
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		NI = problem_size;
		NJ = problem_size;
	}
	std::cout << "Problem size: " << NI << "\n";

	std::vector<DATA_TYPE> A(NI * NJ);
	std::vector<DATA_TYPE> B(NI * NJ);

	init(A.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		conv2D(A.data(), B.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;
		using namespace celerity::access;

		celerity::buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(NI, NJ));
		celerity::buffer<DATA_TYPE, 2> B_buffer(range<2>(NI, NJ));

		celerity::distr_queue queue;
		queue.slow_full_sync();
		double t_start = rtclock();

		queue.submit([=](celerity::handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));
			auto B = B_buffer.get_access<access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

			cgh.parallel_for<class conv2D>(range<2>(NI, NJ), [=, NI_ = NI, NJ_ = NJ](item<2> item) {
				const DATA_TYPE c11 = +0.2, c21 = +0.5, c31 = -0.8;
				const DATA_TYPE c12 = -0.3, c22 = +0.6, c32 = -0.9;
				const DATA_TYPE c13 = +0.4, c23 = +0.7, c33 = +0.10;

				const auto i = item[0];
				const auto j = item[1];

				if((i < NI_ - 1) && (j < NJ_ - 1) && (i > 0) && (j > 0)) {
					B[item] = c11 * A[{(i - 1), (j - 1)}] + c12 * A[{(i + 0), (j - 1)}] + c13 * A[{(i + 1), (j - 1)}] + c21 * A[{(i - 1), (j + 0)}]
					          + c22 * A[{(i + 0), (j + 0)}] + c23 * A[{(i + 1), (j + 0)}] + c31 * A[{(i - 1), (j + 1)}] + c32 * A[{(i + 0), (j + 1)}]
					          + c33 * A[{(i + 1), (j + 1)}];
				}
			});
		});

		queue.slow_full_sync();
		double t_end = rtclock();

		queue.with_master_access([=, &B](celerity::handler& cgh) {
			auto out = B_buffer.get_access<access::mode::read>(cgh, B_buffer.get_range());
			cgh.run([=, &B]() {
				fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

				if(shouldDoCpu()) compareResults(B.data(), out.get_pointer());
			});
		});
	}
}
