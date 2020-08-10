#include <vector>

#include <cstdlib>

#include <celerity/celerity.h>

#include "polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

// Problem size
auto NI = 512;
auto NJ = 512;
auto NK = 512;
auto NL = 512;
auto NM = 512;

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* G, const DATA_TYPE* G_outputFromGpu) {
	int i, j, fail;
	fail = 0;

	for(i = 0; i < NI; i++) {
		for(j = 0; j < NL; j++) {
			if(percentDiff(G[i * NL + j], G_outputFromGpu[i * NL + j]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D) {
	int i, j;

	for(i = 0; i < NI; i++) {
		for(j = 0; j < NK; j++) {
			A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for(i = 0; i < NK; i++) {
		for(j = 0; j < NJ; j++) {
			B[i * NJ + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
		}
	}

	for(i = 0; i < NJ; i++) {
		for(j = 0; j < NM; j++) {
			C[i * NM + j] = ((DATA_TYPE)i * (j + 3)) / NL;
		}
	}

	for(i = 0; i < NM; i++) {
		for(j = 0; j < NL; j++) {
			D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
		}
	}
}

void mm3_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, DATA_TYPE* G) {
	int i, j, k;

	/* E := A*B */
	for(i = 0; i < NI; i++) {
		for(j = 0; j < NJ; j++) {
			E[i * NJ + j] = 0;
			for(k = 0; k < NK; ++k) {
				E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
			}
		}
	}

	/* F := C*D */
	for(i = 0; i < NI; i++) {
		for(j = 0; j < NL; j++) {
			F[i * NL + j] = 0;
			for(k = 0; k < NM; ++k) {
				F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
			}
		}
	}

	/* G := E*F */
	for(i = 0; i < NI; i++) {
		for(j = 0; j < NL; j++) {
			G[i * NL + j] = 0;
			for(k = 0; k < NJ; ++k) {
				G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
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
		NM = problem_size;
	}
	std::cout << "Problem size: " << NI << "\n";

	std::vector<DATA_TYPE> A(NI * NK);
	std::vector<DATA_TYPE> B(NK * NJ);
	std::vector<DATA_TYPE> C(NJ * NM);
	std::vector<DATA_TYPE> D(NM * NL);
	std::vector<DATA_TYPE> E(NI * NJ);
	std::vector<DATA_TYPE> F(NJ * NL);
	std::vector<DATA_TYPE> G(NI * NL);

	init_array(A.data(), B.data(), C.data(), D.data());

	if(shouldDoCpu()){
		double t_start = rtclock();
		mm3_cpu(A.data(), B.data(), C.data(), D.data(), E.data(), F.data(), G.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;
		using namespace celerity::access;

		celerity::buffer<DATA_TYPE, 2> A_buffer(A.data(), range<2>(NI, NK));
		celerity::buffer<DATA_TYPE, 2> B_buffer(B.data(), range<2>(NK, NJ));
		celerity::buffer<DATA_TYPE, 2> C_buffer(C.data(), range<2>(NJ, NM));
		celerity::buffer<DATA_TYPE, 2> D_buffer(D.data(), range<2>(NM, NL));
		celerity::buffer<DATA_TYPE, 2> E_buffer(range<2>(NI, NJ));
		celerity::buffer<DATA_TYPE, 2> F_buffer(range<2>(NJ, NL));
		celerity::buffer<DATA_TYPE, 2> G_buffer(range<2>(NI, NL));

		celerity::distr_queue queue;
		queue.slow_full_sync();
		double t_start = rtclock();

		queue.submit([=](celerity::handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh, slice<2>(1));
			auto B = B_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
			auto E = E_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());

			cgh.parallel_for<class MM1>(E_buffer.get_range(), [=, NK_ = NK](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				DATA_TYPE result = 0;
				for(size_t k = 0; k < NK_; k++) {
					result += A[{i, k}] * B[{k, j}];
				}

				E[item] = result;
			});
		});

		queue.submit([=](celerity::handler& cgh) {
			auto C = C_buffer.get_access<access::mode::read>(cgh, slice<2>(1));
			auto D = D_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
			auto F = F_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());

			cgh.parallel_for<class MM2>(F_buffer.get_range(), [=, NM_ = NM](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				DATA_TYPE result = 0;
				for(size_t k = 0; k < NM_; k++) {
					result += C[{i, k}] * D[{k, j}];
				}

				F[item] = result;
			});
		});


		queue.submit([=](celerity::handler& cgh) {
			auto E = E_buffer.get_access<access::mode::read>(cgh, slice<2>(1));
			auto F = F_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
			auto G = G_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());

			cgh.parallel_for<class MM3>(G_buffer.get_range(), [=, NJ_ = NJ](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				DATA_TYPE result = 0;
				for(size_t k = 0; k < NJ_; k++) {
					result += E[{i, k}] * F[{k, j}];
				}

				G[item] = result;
			});
		});

		queue.slow_full_sync();
		double t_end = rtclock();

		queue.with_master_access([=, &G](celerity::handler& cgh) {
			auto out = G_buffer.get_access<access::mode::read>(cgh, G_buffer.get_range());
			cgh.run([=, &G]() {
				fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

				if(shouldDoCpu())compareResults(G.data(), out.get_pointer());
			});
		});
	}
}
