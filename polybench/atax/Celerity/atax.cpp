#include <vector>

#include <cstdlib>

#include <celerity/celerity.h>

#include "polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Problem size
auto NX = 4096;
auto NY = 4096;

#ifndef M_PI
#define M_PI 3.14159
#endif

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* z, const DATA_TYPE* z_outputFromGpu) {
	int i, fail;
	fail = 0;

	for(i = 0; i < NY; i++) {
		if(percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_array(DATA_TYPE* x, DATA_TYPE* A) {
	int i, j;

	for(i = 0; i < NX; i++) {
		x[i] = i * M_PI;
		for(j = 0; j < NY; j++) {
			A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
		}
	}
}

void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp) {
	int i, j;

	for(i = 0; i < NY; i++) {
		y[i] = 0;
	}

	for(i = 0; i < NX; i++) {
		tmp[i] = 0;

		for(j = 0; j < NY; j++) {
			tmp[i] = tmp[i] + A[i * NY + j] * x[j];
		}

		for(j = 0; j < NY; j++) {
			y[j] = y[j] + A[i * NY + j] * tmp[i];
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		NX = problem_size;
		NY = problem_size;
	}
	std::cout << "Problem size: " << NX << "\n";

	std::vector<DATA_TYPE> A(NX * NY);
	std::vector<DATA_TYPE> x(NY);
	std::vector<DATA_TYPE> y(NY);
	std::vector<DATA_TYPE> tmp(NX);

	init_array(x.data(), A.data());

	if(shouldDoCpu()){
		auto t_start = rtclock();
		atax_cpu(A.data(), x.data(), y.data(), tmp.data());
		auto t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;
		using namespace celerity::access;

		celerity::buffer<DATA_TYPE, 2> A_buffer{A.data(), range<2>(NY, NX)};
		celerity::buffer<DATA_TYPE, 2> x_buffer{x.data(), range<2>(NY, 1)};
		celerity::buffer<DATA_TYPE, 2> y_buffer{range<2>(NY, 1)};
		celerity::buffer<DATA_TYPE, 2> tmp_buffer{range<2>(NX, 1)};

		celerity::distr_queue queue;
		queue.slow_full_sync();
		double t_start = rtclock();

		queue.submit([=](celerity::handler& cgh) {
			auto y = y_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());
			cgh.parallel_for<class Atax1>(y_buffer.get_range(), [=](item<2> item) { y[item] = 0; });
		});

		queue.submit([=](celerity::handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh, slice<2>(1));
			auto x = x_buffer.get_access<access::mode::read>(cgh, celerity::access::all<2>());
			auto tmp = tmp_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());
			cgh.parallel_for<class Atax2>(tmp_buffer.get_range(), [=, NY_ = NY](item<2> item) {
				const auto i = item[0];

				DATA_TYPE result = 0;
				for(size_t j = 0; j < NY_; j++) {
					result += A[{i, j}] * x[{j, 0}];
				}
				tmp[item] = result;
			});
		});

		queue.submit([=](celerity::handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
			auto tmp = tmp_buffer.get_access<access::mode::read>(cgh, celerity::access::all<2>());
			auto y = y_buffer.get_access<access::mode::discard_write>(cgh, one_to_one<2>());
			cgh.parallel_for<class Atax3>(y_buffer.get_range(), [=, NX_ = NX](item<2> item) {
				const auto j = item[0];

				DATA_TYPE result = 0;
				for(size_t i = 0; i < NX_; i++) {
					result += A[{i, j}] * tmp[{i, 0}];
				}
				y[item] = result;
			});
		});

		queue.slow_full_sync();
		double t_end = rtclock();

		queue.with_master_access([=, &y](celerity::handler& cgh) {
			auto out = y_buffer.get_access<access::mode::read>(cgh, y_buffer.get_range());
			cgh.run([=, &y]() {
				fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

				if(shouldDoCpu())compareResults(y.data(), out.get_pointer());
			});
		});
	}
}
