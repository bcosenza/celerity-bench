#include <iostream>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <celerity/celerity.h>

#include "polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Problem size
auto M = 2048;
auto N = 2048;

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* A, const DATA_TYPE* A_outputFromGpu) {
	int i, j, fail;
	fail = 0;

	for(i = 0; i < M; i++) {
		for(j = 0; j < N; j++) {
			if(percentDiff(A[i * N + j], A_outputFromGpu[i * N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
				fail++;
				// printf("i: %d j: %d \n1: %f\n 2: %f\n", i, j, A[i * N + j], A_outputFromGpu[i * N + j]);
			}
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_array(DATA_TYPE* A) {
	int i, j;

	for(i = 0; i < M; i++) {
		for(j = 0; j < N; j++) {
			A[i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
		}
	}
}

void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q) {
	int i, j, k;
	DATA_TYPE nrm;
	for(k = 0; k < N; k++) {
		nrm = 0;
		for(i = 0; i < M; i++) {
			nrm += A[i * N + k] * A[i * N + k];
		}

		R[k * N + k] = sqrt(nrm);
		for(i = 0; i < M; i++) {
			Q[i * N + k] = A[i * N + k] / R[k * N + k];
		}

		for(j = k + 1; j < N; j++) {
			R[k * N + j] = 0;
			for(i = 0; i < M; i++) {
				R[k * N + j] += Q[i * N + k] * A[i * N + j];
			}
			for(i = 0; i < M; i++) {
				A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
			}
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		M = problem_size;
		N = problem_size;
	}
	std::cout << "Problem size: " << M << "\n";

	std::vector<DATA_TYPE> A(M * N);
	std::vector<DATA_TYPE> R(M * N);
	std::vector<DATA_TYPE> Q(M * N);

	init_array(A.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		gramschmidt(A.data(), R.data(), Q.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;
		using namespace celerity::access;

		std::vector<DATA_TYPE> A_gpu_init(M * N);
		init_array(A_gpu_init.data());

		celerity::buffer<DATA_TYPE, 2> A_buffer{A_gpu_init.data(), cl::sycl::range<2>(M, N)};
		celerity::buffer<DATA_TYPE, 2> R_buffer{range<2>(M, N)};
		celerity::buffer<DATA_TYPE, 2> Q_buffer{range<2>(M, N)};

		celerity::distr_queue queue;
		queue.slow_full_sync();
		double t_start = rtclock();

		for(size_t k = 0; k < N; k++) {
			const auto non_empty_chunk_range_mapper = [=](celerity::chunk<2> chunk) -> celerity::subrange<2> {
				if(chunk.range[0] == 0) return {};
				return {{k, k}, {1, 1}};
			};

			queue.submit([=](celerity::handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
				auto R = R_buffer.get_access<access::mode::discard_write>(cgh, non_empty_chunk_range_mapper);

				// TODO: use reduction
				cgh.parallel_for<class Gramschmidt1>(range<2>(1, 1), [=, M_ = M](item<2> item) {
					DATA_TYPE nrm = 0;
					for(size_t i = 0; i < M_; i++) {
						nrm += A[{i, k}] * A[{i, k}];
					}
					R[{k, k}] = sqrt(nrm);
				});
			});

			queue.submit([=](celerity::handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read>(cgh, one_to_one<2>());
				auto R = R_buffer.get_access<access::mode::read>(cgh, celerity::access::fixed<2>({{k, k}, {1, 1}}));
				auto Q = Q_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());
				cgh.parallel_for<class Gramschmidt2>(range<2>(M, 1), id<2>(0, k), [=](item<2> item) { Q[item] = A[item] / R[{k, k}]; });
			});

			queue.submit([=](celerity::handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read_write>(cgh, slice<2>(0));
				auto R = R_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());
				auto Q = Q_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
				cgh.parallel_for<class Gramschmidt3>(range<2>(1, N - k - 1), id<2>(k, k + 1), [=, M_ = M](item<2> item) {
					const auto k = item[0];
					const auto j = item[1];

					DATA_TYPE R_result = 0;
					for(size_t i = 0; i < M_; i++) {
						R_result += Q[{i, k}] * A[{i, j}];
					}
					R_result;

					for(size_t i = 0; i < M_; i++) {
						A[{i, j}] -= Q[{i, k}] * R_result;
					}

					R[{k, j}] = R_result;
				});
			});
		}

		queue.slow_full_sync();
		double t_end = rtclock();

		queue.with_master_access([=, &A](celerity::handler& cgh) {
			auto A_gpu = A_buffer.get_access<access::mode::read>(cgh, A_buffer.get_range());
			cgh.run([=, &A]() {
				fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

				if(shouldDoCpu()) compareResults(A.data(), A_gpu.get_pointer());
			});
		});
	}
}
