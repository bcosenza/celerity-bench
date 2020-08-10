#include <vector>

#include <cstdlib>

#include <celerity/celerity.h>

#include "polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

using DATA_TYPE = float;

// Problem size
int NI = 256;
int NJ = 256;
int NK = 256;

void compareResults(const DATA_TYPE* B, const DATA_TYPE* B_outputFromGpu) {
	int i, j, k, fail;
	fail = 0;

	// Compare result from cpu and gpu...
	for(i = 1; i < NI - 1; ++i) {
		for(j = 1; j < NJ - 1; ++j) {
			for(k = 1; k < NK - 1; ++k) {
				if(percentDiff(B[i * (NK * NJ) + j * NK + k], B_outputFromGpu[i * (NK * NJ) + j * NK + k]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
			}
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init(DATA_TYPE* A) {
	int i, j, k;

	for(i = 1; i < NI - 1; ++i) {
		for(j = 1; j < NJ - 1; ++j) {
			for(k = 1; k < NK - 1; ++k) {
				A[i * (NK * NJ) + j * NK + k] = (float)rand() / RAND_MAX;
			}
		}
	}
}

void conv3D(DATA_TYPE* A, DATA_TYPE* B) {
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	// clang-format off
	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;
	// clang-format on

	for(i = 1; i < NI - 1; ++i) {
		for(j = 1; j < NJ - 1; ++j) {
			for(k = 1; k < NK - 1; ++k) {
				B[i * (NK * NJ) + j * NK + k] = c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)]
				                                + c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)]
				                                + c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)]
				                                + c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] + c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)]
				                                + c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] + c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)]
				                                + c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)]
				                                + c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)]
				                                + c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
			}
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		NI = problem_size;
		NJ = problem_size;
		NK = problem_size;
	}
	std::cout << "Problem size: " << NI << "\n";

	std::vector<DATA_TYPE> A(NI * NJ * NK);
	std::vector<DATA_TYPE> B(NI * NJ * NK);

	init(A.data());

	if(shouldDoCpu()) {
		auto t_start = rtclock();
		conv3D(A.data(), B.data());
		auto t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;
		using namespace celerity::access;

		celerity::buffer<DATA_TYPE, 3> A_buffer(A.data(), range<3>(NI, NJ, NK));
		celerity::buffer<DATA_TYPE, 3> B_buffer(B.data(), range<3>(NI, NJ, NK));

		celerity::distr_queue queue;
		queue.slow_full_sync();
		auto t_start = rtclock();

		for(size_t i = 1; i < NI - 1; i++) {
			queue.submit([=](celerity::handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read>(cgh, celerity::access::neighborhood<3>(1, 1, 1));
				auto B = B_buffer.get_access<access::mode::discard_write>(cgh, celerity::access::one_to_one<3>());

				cgh.parallel_for<class conv3D>(range<3>(1, NJ, NK), [=, NI_ = NI, NJ_ = NJ, NK_ = NK](item<3> item) {
					const DATA_TYPE c11 = +2, c21 = +5, c31 = -8;
					const DATA_TYPE c12 = -3, c22 = +6, c32 = -9;
					const DATA_TYPE c13 = +4, c23 = +7, c33 = +10;

					const auto j = item[1];
					const auto k = item[2];

					if((i < (NI_ - 1)) && (j < (NJ_ - 1)) && (k < (NK_ - 1)) && (i > 0) && (j > 0) && (k > 0)) {
						B[item] = c11 * A[{(i - 1), (j - 1), (k - 1)}] + c13 * A[{(i + 1), (j - 1), (k - 1)}] + c21 * A[{(i - 1), (j - 1), (k - 1)}]
						          + c23 * A[{(i + 1), (j - 1), (k - 1)}] + c31 * A[{(i - 1), (j - 1), (k - 1)}] + c33 * A[{(i + 1), (j - 1), (k - 1)}]
						          + c12 * A[{(i + 0), (j - 1), (k + 0)}] + c22 * A[{(i + 0), (j + 0), (k + 0)}] + c32 * A[{(i + 0), (j + 1), (k + 0)}]
						          + c11 * A[{(i - 1), (j - 1), (k + 1)}] + c13 * A[{(i + 1), (j - 1), (k + 1)}] + c21 * A[{(i - 1), (j + 0), (k + 1)}]
						          + c23 * A[{(i + 1), (j + 0), (k + 1)}] + c31 * A[{(i - 1), (j + 1), (k + 1)}] + c33 * A[{(i + 1), (j + 1), (k + 1)}];
					} else {
						B[item] = 0;
					}
				});
			});
		}

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
