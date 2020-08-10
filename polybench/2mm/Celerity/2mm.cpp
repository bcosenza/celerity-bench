#include <vector>

#include <celerity/celerity.h>

#include "polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

// Problem size.
auto NI = 2048;
auto NJ = 2048;
auto NK = 2048;
auto NL = 2048;

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* E, const DATA_TYPE* E_outputFromGpu) {
	int i, j, fail;
	fail = 0;

	for(i = 0; i < NL; i++) {
		for(j = 0; j < NI; j++) {
			if(percentDiff(E[i * NI + j], E_outputFromGpu[i * NI + j]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D) {
	int i, j;

	for(i = 0; i < NI; i++) {
		for(j = 0; j < NK; j++) {
			A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for(i = 0; i < NK; i++) {
		for(j = 0; j < NJ; j++) {
			B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
		}
	}

	for(i = 0; i < NL; i++) {
		for(j = 0; j < NJ; j++) {
			C[i * NL + j] = ((DATA_TYPE)i * (j + 3)) / NL;
		}
	}

	for(i = 0; i < NI; i++) {
		for(j = 0; j < NL; j++) {
			D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
		}
	}
}

void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E) {
	int i, j, k;

	for(i = 0; i < NI; i++) {
		for(j = 0; j < NJ; j++) {
			for(k = 0; k < NK; ++k) {
				C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
			}
		}
	}

	for(i = 0; i < NI; i++) {
		for(j = 0; j < NL; j++) {
			for(k = 0; k < NJ; ++k) {
				E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
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
		NL = problem_size;
	}
	std::cout << "Problem size: " << NI << "\n";

	std::vector<DATA_TYPE> A(NI * NK);
	std::vector<DATA_TYPE> B(NK * NJ);
	std::vector<DATA_TYPE> C(NI * NJ);
	std::vector<DATA_TYPE> D(NJ * NL);
	std::vector<DATA_TYPE> E(NI * NL);

	init_array(A.data(), B.data(), C.data(), D.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		mm2_cpu(A.data(), B.data(), C.data(), D.data(), E.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;
		using namespace celerity::access;

		celerity::buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(NK, NI));
		celerity::buffer<DATA_TYPE, 2> B_buffer(B.data(), range<2>(NJ, NK));
		celerity::buffer<DATA_TYPE, 2> C_buffer(C.data(), range<2>(NJ, NI));
		celerity::buffer<DATA_TYPE, 2> D_buffer(D.data(), range<2>(NL, NJ));
		celerity::buffer<DATA_TYPE, 2> E_buffer(range<2>(NL, NI));

		celerity::distr_queue queue;
		queue.slow_full_sync();
		double t_start = rtclock();

		queue.submit([=](celerity::handler& cgh) {
			auto A_access = A_buffer.get_access<access::mode::read>(cgh, slice<2>(1));
			auto B_access = B_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
			auto C_access = C_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());

			cgh.parallel_for<class MM1>(C_buffer.get_range(), [=, NK_ = NK](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				DATA_TYPE result = 0;
				for(size_t k = 0; k < NK_; k++) {
					result += A_access[{i, k}] * B_access[{k, j}];
				}

				C_access[item] = result;
			});
		});

		queue.submit([=](celerity::handler& cgh) {
			auto C_access = C_buffer.get_access<access::mode::read>(cgh, slice<2>(1));
			auto D_access = D_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
			auto E_access = E_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());

			cgh.parallel_for<class MM2>(E_buffer.get_range(), [=, NJ_ = NJ](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				DATA_TYPE result = 0;
				for(size_t k = 0; k < NJ_; k++) {
					result += C_access[{i, k}] * D_access[{k, j}];
				}
				E_access[item] = result;
			});
		});

		queue.slow_full_sync();
		double t_end = rtclock();

		queue.with_master_access([=, &E](celerity::handler& cgh) {
			auto out = E_buffer.get_access<access::mode::read>(cgh, E_buffer.get_range());
			cgh.run([=, &E]() {
				fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
				if(shouldDoCpu()) compareResults(E.data(), out.get_pointer());
			});
		});
	}
}
