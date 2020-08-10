#include <iostream>
#include <stdexcept>
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

void compareResults(const DATA_TYPE* s, const DATA_TYPE* s_outputFromGpu, const DATA_TYPE* q, const DATA_TYPE* q_outputFromGpu) {
	int i, fail;
	fail = 0;

	for(i = 0; i < NX; i++) {
		if(percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
	}

	for(i = 0; i < NY; i++) {
		if(percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_array(DATA_TYPE* A, DATA_TYPE* p, DATA_TYPE* r) {
	int i, j;

	for(i = 0; i < NX; i++) {
		r[i] = i * M_PI;

		for(j = 0; j < NY; j++) {
			A[i * NY + j] = ((DATA_TYPE)i * j) / NX;
		}
	}

	for(i = 0; i < NY; i++) {
		p[i] = i * M_PI;
	}
}

void bicg_cpu(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q) {
	int i, j;

	for(i = 0; i < NY; i++) {
		s[i] = 0.0;
	}

	for(i = 0; i < NX; i++) {
		q[i] = 0.0;
		for(j = 0; j < NY; j++) {
			s[j] = s[j] + r[i] * A[i * NY + j];
			q[i] = q[i] + A[i * NY + j] * p[j];
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
	std::vector<DATA_TYPE> r(NX);
	std::vector<DATA_TYPE> s(NY);
	std::vector<DATA_TYPE> p(NY);
	std::vector<DATA_TYPE> q(NX);

	init_array(A.data(), p.data(), r.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		bicg_cpu(A.data(), r.data(), s.data(), p.data(), q.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;
		using namespace celerity::access;

		celerity::buffer<DATA_TYPE, 2> A_buffer{A.data(), range<2>(NY, NX)};
		celerity::buffer<DATA_TYPE, 2> r_buffer{r.data(), range<2>(NX, 1)};
		celerity::buffer<DATA_TYPE, 2> p_buffer{p.data(), range<2>(NY, 1)};
		celerity::buffer<DATA_TYPE, 2> q_buffer{range<2>(NX, 1)};
		celerity::buffer<DATA_TYPE, 2> s_buffer{range<2>(NY, 1)};

		// for simplicity:
		if(NX != NY) throw std::runtime_error("NX != NY");

		celerity::distr_queue queue;
		queue.slow_full_sync();
		double t_start = rtclock();

		queue.submit([=](celerity::handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh, slice<2>(1));
			auto p = p_buffer.get_access<access::mode::read>(cgh, celerity::access::all<2>());
			auto q = q_buffer.get_access<access::mode::write>(cgh, one_to_one<2>());

			cgh.parallel_for<class Bicg1>(q_buffer.get_range(), [=, NY_ = NY](item<2> item) {
				const auto i = item[0];

				DATA_TYPE result = 0;
				for(size_t j = 0; j < NY_; j++) {
					result += A[{i, j}] * p[{j, 0}];
				}
				q[item] = result;
			});
		});

		queue.submit([=](celerity::handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh, slice<2>(0));
			auto r = r_buffer.get_access<access::mode::read>(cgh, celerity::access::all<2>());
			auto s = s_buffer.get_access<access::mode::write>(cgh, one_to_one<2>());

			cgh.parallel_for<class Bicg2>(s_buffer.get_range(), [=, NX_ = NX](item<2> item) {
				const auto j = item[0];

				DATA_TYPE result = 0;
				for(size_t i = 0; i < NX_; i++) {
					result += r[{i, 0}] * A[{i, j}];
				}
				s[item] = result;
			});
		});

		queue.slow_full_sync();
		double t_end = rtclock();

		queue.with_master_access([=, &q, &s](celerity::handler& cgh) {
			auto s_gpu = s_buffer.get_access<access::mode::read>(cgh, s_buffer.get_range());
			auto q_gpu = q_buffer.get_access<access::mode::read>(cgh, q_buffer.get_range());
			cgh.run([=, &q, &s]() {
				fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

				if(shouldDoCpu()) compareResults(s.data(), s_gpu.get_pointer(), q.data(), q_gpu.get_pointer());
			});
		});
	}
}
