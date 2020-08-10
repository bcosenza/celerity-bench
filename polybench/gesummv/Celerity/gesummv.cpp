#include <iostream>
#include <vector>

#include <cstdlib>

#include <celerity/celerity.h>

#include "polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Problem size
auto N = 4096;

using DATA_TYPE = float;

constexpr DATA_TYPE ALPHA = 1;
constexpr DATA_TYPE BETA = 1;

void compareResults(const DATA_TYPE* y, const DATA_TYPE* y_outputFromGpu) {
	int i, fail;
	fail = 0;

	for(i = 0; i < (N); i++) {
		if(percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x) {
	int i, j;

	for(i = 0; i < N; i++) {
		x[i] = ((DATA_TYPE)i) / N;

		for(j = 0; j < N; j++) {
			A[i * N + j] = ((DATA_TYPE)i * j) / N;
			B[i * N + j] = ((DATA_TYPE)i * j) / N;
		}
	}
}

void gesummv(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp) {
	int i, j;

	for(i = 0; i < N; i++) {
		tmp[i] = 0;
		y[i] = 0;
		for(j = 0; j < N; j++) {
			tmp[i] = A[i * N + j] * x[j] + tmp[i];
			y[i] = B[i * N + j] * x[j] + y[i];
		}

		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		N = problem_size;
	}
	std::cout << "Problem size: " << N << "\n";

	std::vector<DATA_TYPE> A(N * N);
	std::vector<DATA_TYPE> B(N * N);
	std::vector<DATA_TYPE> x(N);
	std::vector<DATA_TYPE> y(N);
	std::vector<DATA_TYPE> tmp(N);

	init(A.data(), B.data(), x.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		gesummv(A.data(), B.data(), x.data(), y.data(), tmp.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;
		using namespace celerity::access;

		celerity::buffer<DATA_TYPE, 2> A_buffer{A.data(), range<2>(N, N)};
		celerity::buffer<DATA_TYPE, 2> B_buffer{B.data(), range<2>(N, N)};
		celerity::buffer<DATA_TYPE, 1> x_buffer{x.data(), range<1>(N)};
		celerity::buffer<DATA_TYPE, 1> y_buffer{range<1>(N)};
		celerity::buffer<DATA_TYPE, 1> tmp_buffer{range<1>(N)};

		celerity::distr_queue queue;
		queue.slow_full_sync();
		double t_start = rtclock();

		queue.submit([=](celerity::handler& cgh) {
			// TODO: Upstream this to Celerity runtime:
			const auto rows = [=](celerity::chunk<1> chunk) -> celerity::subrange<2> {
				return {{chunk.offset[0], 0}, {chunk.range[0], (size_t)N}};
			};

			auto A = A_buffer.get_access<access::mode::read>(cgh, rows);
			auto B = B_buffer.get_access<access::mode::read>(cgh, rows);
			auto x = x_buffer.get_access<access::mode::read>(cgh, one_to_one<1>());
			auto y = y_buffer.get_access<access::mode::read_write>(cgh, one_to_one<1>());
			auto tmp = tmp_buffer.get_access<access::mode::read_write>(cgh, one_to_one<1>());
			cgh.parallel_for<class Gesummv>(y.get_range(), [=, N_ = N](item<1> item) {
				const auto i = item[0];

				DATA_TYPE tmp_result = tmp[item];
				DATA_TYPE y_result = y[item];

				for(size_t j = 0; j < N_; j++) {
					tmp_result += A[{i, j}] * x[j];
					y_result += B[{i, j}] * x[j];
				}

				tmp[item] = tmp_result;
				y[item] = ALPHA * tmp_result + BETA * y_result;
			});
		});

		queue.slow_full_sync();
		double t_end = rtclock();

		queue.with_master_access([=, &y](celerity::handler& cgh) {
			auto y_gpu = y_buffer.get_access<access::mode::read>(cgh, y_buffer.get_range());
			cgh.run([=, &y]() {
				fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
				if(shouldDoCpu()) compareResults(y.data(), y_gpu.get_pointer());
			});
		});
	}
}
