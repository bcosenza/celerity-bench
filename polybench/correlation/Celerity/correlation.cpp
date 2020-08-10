#include <iostream>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <celerity/celerity.h>

#include "polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

// Problem size
auto M = 2048;
auto N = 2048;

#define FLOAT_N 3214212.01
#define EPS 0.005

#define sqrt_of_array_cell(x, j) sqrt(x[j])

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* symmat, const DATA_TYPE* symmat_outputFromGpu) {
	int i, j, fail;
	fail = 0;

	for(i = 0; i <= 0; i++) {
		for(j = 0; j <= N; j++) {
			if(percentDiff(symmat[i * (N + 1) + j], symmat_outputFromGpu[i * (N + 1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
				fail++;
				printf("I: %d J: %d \n 1: %f\n 2: %f\n", i, j, symmat[i * (N + 1) + j], symmat_outputFromGpu[i * (N + 1) + j]);
			}
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_arrays(DATA_TYPE* data) {
	int i, j;

	for(i = 0; i <= M; i++) {
		for(j = 0; j <= N; j++) {
			data[i * N + j] = ((DATA_TYPE)i * j) / (M + 1);
		}
	}
}

void correlation(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat) {
	int i, j, j1, j2;

	// Determine mean of column vectors of input data matrix
	for(j = 1; j <= M; j++) {
		mean[j] = 0.0;

		for(i = 1; i <= N; i++) {
			mean[j] += data[i * (M + 1) + j];
		}

		mean[j] /= (DATA_TYPE)FLOAT_N;
	}

	// Determine standard deviations of column vectors of data matrix.
	for(j = 1; j <= M; j++) {
		stddev[j] = 0.0;

		for(i = 1; i <= N; i++) {
			stddev[j] += (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
		}

		stddev[j] /= FLOAT_N;
		stddev[j] = sqrt_of_array_cell(stddev, j);
		stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
	}

	// Center and reduce the column vectors.
	for(i = 1; i <= N; i++) {
		for(j = 1; j <= M; j++) {
			data[i * (M + 1) + j] -= mean[j];
			data[i * (M + 1) + j] /= sqrt(FLOAT_N);
			data[i * (M + 1) + j] /= stddev[j];
		}
	}

	// Calculate the m * m correlation matrix.
	for(j1 = 1; j1 <= M - 1; j1++) {
		symmat[j1 * (M + 1) + j1] = 1.0;

		for(j2 = j1 + 1; j2 <= M; j2++) {
			symmat[j1 * (M + 1) + j2] = 0.0;

			for(i = 1; i <= N; i++) {
				symmat[j1 * (M + 1) + j2] += (data[i * (M + 1) + j1] * data[i * (M + 1) + j2]);
			}

			symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
		}
	}

	symmat[M * (M + 1) + M] = 1.0;
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		M = problem_size;
		N = problem_size;
	}
	std::cout << "Problem size: " << M << "\n";

	std::vector<DATA_TYPE> data((M + 1) * (N + 1));
	std::vector<DATA_TYPE> mean(M + 1);
	std::vector<DATA_TYPE> stddev(M + 1);
	std::vector<DATA_TYPE> symmat((M + 1) * (N + 1));

	init_arrays(data.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		correlation(data.data(), mean.data(), stddev.data(), symmat.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;
		using namespace celerity::access;

		std::vector<DATA_TYPE> data_gpu((M + 1) * (N + 1));
		std::vector<DATA_TYPE> mean_gpu(M + 1);
		std::vector<DATA_TYPE> stddev_gpu(M + 1);
		std::vector<DATA_TYPE> symmat_gpu((M + 1) * (N + 1));

		init_arrays(data_gpu.data());

		celerity::buffer<DATA_TYPE, 2> data_buffer{data_gpu.data(), range<2>(M + 1, N + 1)};
		celerity::buffer<DATA_TYPE, 2> mean_buffer{mean_gpu.data(), range<2>(M + 1, 1)};
		celerity::buffer<DATA_TYPE, 2> stddev_buffer{stddev_gpu.data(), range<2>(M + 1, 1)};
		celerity::buffer<DATA_TYPE, 2> symmat_buffer{symmat_gpu.data(), range<2>(M + 1, N + 1)};

		celerity::distr_queue queue;
		queue.slow_full_sync();
		double t_start = rtclock();

		queue.submit([=](celerity::handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
			auto mean = mean_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());
			cgh.parallel_for<class Correlation1>(range<2>(M, 1), id<2>(1, 0), [=, N_ = N](item<2> item) {
				const auto j = item[0];

				DATA_TYPE result = 0;
				for(size_t i = 1; i <= N_; i++) {
					result += data[{i, j}];
				}
				mean[item] = result / ((DATA_TYPE)FLOAT_N);
			});
		});

		queue.submit([=](celerity::handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
			auto mean = mean_buffer.get_access<access::mode::read>(cgh, one_to_one<2>());
			auto stddev = stddev_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());
			cgh.parallel_for<class Correlation2>(range<2>(M, 1), id<2>(1, 0), [=, N_ = N](item<2> item) {
				const auto j = item[0];

				DATA_TYPE result = 0;
				for(size_t i = 1; i <= N_; i++) {
					result += (data[{i, j}] - mean[item]) * (data[{i, j}] - mean[item]);
				}
				result /= FLOAT_N;
				result = sqrt(result);

				stddev[item] = result <= EPS ? 1.0 : result;
			});
		});

		queue.submit([=](celerity::handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read_write>(cgh, one_to_one<2>());
			auto mean = mean_buffer.get_access<access::mode::read>(cgh, celerity::access::all<2>());
			auto stddev = stddev_buffer.get_access<access::mode::read>(cgh, celerity::access::all<2>());
			cgh.parallel_for<class Correlation3>(range<2>(M, N), id<2>(1, 1), [=](item<2> item) {
				const auto j = item[1];

				auto result = data[item];
				result -= mean[{j, 0}];
				result /= sqrt(FLOAT_N);
				result /= stddev[{j, 0}];

				data[item] = result;
			});
		});

		queue.submit([=](celerity::handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh, celerity::access::all<2>());
			auto symmat = symmat_buffer.get_access<access::mode::discard_write>(cgh, slice<2>(1));
			auto symmat2 = symmat_buffer.get_access<access::mode::discard_write>(cgh, slice<2>(0));
			cgh.parallel_for<class Correlation4>(range<2>(M - 1, 1), id<2>(1, 0), [=, M_ = M, N_ = N](item<2> item) {
				const auto j1 = item[0];

				symmat[{j1, j1}] = 1.0;

				for(size_t j2 = j1 + 1; j2 <= M_; j2++) {
					DATA_TYPE result = 0.0;
					for(size_t i = 1; i <= N_; i++) {
						result += data[{i, j1}] * data[{i, j2}];
					}

					symmat[{j1, j2}] = result;
					symmat2[{j2, j1}] = result;
				}
			});
		});

		queue.submit([=](celerity::handler& cgh) {
			auto symmat = symmat_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());
			cgh.parallel_for<class Correlation5>(range<2>(1, 1), id<2>(M, M), [=](item<2> item) { symmat[item] = 1.0; });
		});

		queue.slow_full_sync();
		double t_end = rtclock();

		queue.with_master_access([=, &symmat](celerity::handler& cgh) {
			auto symmat_gpu = symmat_buffer.get_access<access::mode::read>(cgh, symmat_buffer.get_range());
			cgh.run([=, &symmat]() {
				fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
				if(shouldDoCpu()) compareResults(symmat.data(), symmat_gpu.get_pointer());
			});
		});
	}
}
