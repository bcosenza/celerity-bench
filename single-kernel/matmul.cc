#include <cstdio>
#include <vector>

#include <common.h>
#include <celerity.h>

// Performs matrix multiply

class Matmul;

// template <typename T>
void set_identity(celerity::distr_queue queue, celerity::buffer<BENCH_DATA_TYPE, 2> mat) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor dw{mat, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class set_identity_kernel>(mat.get_range(), [=](celerity::item<2> item) { dw[item] = item[0] == item[1]; });
	});
}

//template <typename T>
void multiply(celerity::distr_queue queue, celerity::buffer<BENCH_DATA_TYPE, 2> mat_a, celerity::buffer<BENCH_DATA_TYPE, 2> mat_b, celerity::buffer<BENCH_DATA_TYPE, 2> mat_c, const size_t mat_size) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor a{mat_a, cgh, celerity::access::slice<2>(1), celerity::read_only};
		celerity::accessor b{mat_b, cgh, celerity::access::slice<2>(0), celerity::read_only};
		celerity::accessor c{mat_c, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};

/* #if CELERITY_FEATURE_LOCAL_ACCESSOR

		// Use local-memory tiling to avoid waiting on global memory too often
		const size_t GROUP_SIZE = 8;
		celerity::local_accessor<BENCH_DATA_TYPE, 2> scratch_a{{GROUP_SIZE, GROUP_SIZE}, cgh};
		celerity::local_accessor<BENCH_DATA_TYPE, 2> scratch_b{{GROUP_SIZE, GROUP_SIZE}, cgh};

		cgh.parallel_for<class Matmul>(celerity::nd_range<2>{{mat_size, mat_size}, {GROUP_SIZE, GROUP_SIZE}}, [=](celerity::nd_item<2> item) {
			BENCH_DATA_TYPE sum{};
			const auto lid = item.get_local_id();
			for(size_t j = 0; j < mat_size; j += GROUP_SIZE) {
				scratch_a[lid] = a[item.get_group(0) * GROUP_SIZE + lid[0]][j + lid[1]];
				scratch_b[lid] = b[j + lid[0]][item.get_group(1) * GROUP_SIZE + lid[1]];
				celerity::group_barrier(item.get_group());

				for(size_t k = 0; k < GROUP_SIZE; ++k) {
					const auto a_ik = scratch_a[lid[0]][k];
					const auto b_kj = scratch_b[k][lid[1]];
					sum += a_ik * b_kj;
				}
				celerity::group_barrier(item.get_group());
			}
			c[item.get_global_id()] = sum;
		});

#else */

		cgh.parallel_for<class Matmul>(celerity::range<2>(mat_size, mat_size), [=](celerity::item<2> item) {
			BENCH_DATA_TYPE sum{};
			for(size_t k = 0; k < mat_size; ++k) {
				const auto a_ik = a[{item[0], k}];
				const auto b_kj = b[{k, item[1]}];
				sum += a_ik * b_kj;
			}
			c[item] = sum;
		});

// #endif
	});
}


class Matmul {
protected:    
	//std::vector<BENCH_DATA_TYPE> mat_a;
	//std::vector<BENCH_DATA_TYPE> mat_b;
	//std::vector<BENCH_DATA_TYPE> mat_res;
	BenchmarkArgs args;
	int mat_size;

  PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
  PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_b_buf;
  PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_res_buf;

  //celerity::range<2> mrange;
  //celerity::buffer<BENCH_DATA_TYPE, 2> mat_a_buf(celerity::range<2>(1024, 1024));
  //celerity::buffer<BENCH_DATA_TYPE, 2> mat_b_buf(celerity::range<2>(1024, 1024));
  //celerity::buffer<BENCH_DATA_TYPE, 2> mat_c_buf(celerity::range<2>(1024, 1024));

public:
	Matmul(const BenchmarkArgs &_args) : args(_args) {
    mat_size = args.problem_size;
	//Matmul(int problem_size) {
		//mat_size = problem_size;
	}

	//void setup(celerity::distr_queue queue) {
	void setup() {
		/*mat_a = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
		mat_b = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);
		mat_res = std::vector<BENCH_DATA_TYPE>(mat_size * mat_size);

		// Initialize matrices to the identity
		for(size_t i = 0; i < mat_size; ++i) {
			for(size_t j = 0; j < mat_size; ++j) {
				mat_a[i * mat_size + j] = i == j;
				mat_b[i * mat_size + j] = i == j;
			}
		}*/

		//mat_a_buf.initialize(mat_a.data(), celerity::range<2>(mat_size, mat_size));
		//mat_b_buf.initialize(mat_b.data(), celerity::range<2>(mat_size, mat_size));
		auto range = celerity::range<2>(mat_size, mat_size);
    //  mat_a_buf.initialize  (celerity::range<2>(mat_size, mat_size));
    //  mat_b_buf.initialize  (celerity::range<2>(mat_size, mat_size));
		//  mat_res_buf.initialize(celerity::range<2>(mat_size, mat_size));
    mat_a_buf.initialize(range);
    mat_b_buf.initialize(range);
		mat_res_buf.initialize(range);
    set_identity(QueueManager::getInstance(), mat_a_buf.get());
    set_identity(QueueManager::getInstance(), mat_b_buf.get());
	}

	//void run(celerity::distr_queue queue, celerity::buffer<BENCH_DATA_TYPE, 2> mat_a_buf, celerity::buffer<BENCH_DATA_TYPE, 2> mat_b_buf,
    //celerity::buffer<BENCH_DATA_TYPE, 2> mat_res_buf) {
	void run() {
    //multiply(queue, mat_a_buf, mat_b_buf, mat_res_buf, mat_size);
    multiply(QueueManager::getInstance(), mat_a_buf.get(), mat_b_buf.get(), mat_res_buf.get(), mat_size);
	}

	static std::string getBenchmarkName() { return "Matmul"; }

	bool verify(VerificationSetting &ver) {
		bool verification_passed = true;
		/*auto range = celerity::range<2>(mat_size, mat_size);

		QueueManager::getInstance().submit(celerity::allow_by_ref, [&](celerity::handler& cgh) {
			//auto result = mat_res_buf.template get_access<cl::sycl::access::mode::read>(cgh, celerity::range<2>(mat_size, mat_size));
			celerity::accessor result{mat_res_buf.get(), cgh, celerity::access::one_to_one{}, celerity::read_only_host_task};
			//auto result{mat_res_buf, cgh, celerity::access::one_to_one{}, celerity::read_only_host_task};

			cgh.host_task(mat_res_buf.get().get_range(), [=, &verification_passed](celerity::partition<2> part) {
			//cgh.host_task(range, [=, &verification_passed]() {

				for(size_t i = 0; i < mat_size; ++i) {
					for(size_t j = 0; j < mat_size; ++j) {
						const float kernel_value = result[{i, j}];
						const float host_value = i == j;
						if(kernel_value != host_value) {
							//fprintf(stderr, "VERIFICATION FAILED for element %ld,%ld: %f != %f\n", i, j, kernel_value, host_value);
							verification_passed = false;
							break;
						}
					}
					if(!verification_passed) { break; }
				}
			});
		});*/
		QueueManager::sync();
		return verification_passed;
	}
};

int main(int argc, char** argv) {
  /*//celerity::distr_queue queue;
  int mat_size = 1024;
  auto range = celerity::range<2>(mat_size, mat_size);
  //celerity::buffer<BENCH_DATA_TYPE, 2> mat_a_buf(range);
  //celerity::buffer<BENCH_DATA_TYPE, 2> mat_b_buf(range);
  //celerity::buffer<BENCH_DATA_TYPE, 2> mat_res_buf(range);
  PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_a_buf;
  PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_b_buf;
  PrefetchedBuffer<BENCH_DATA_TYPE, 2> mat_res_buf;
  Matmul matmul(mat_size);
    mat_a_buf.initialize(range);
    mat_b_buf.initialize(range);
		mat_res_buf.initialize(range);
    set_identity(QueueManager::getInstance(), mat_a_buf.get());
    set_identity(QueueManager::getInstance(), mat_b_buf.get());
  //set_identity(queue, mat_a_buf);
  //set_identity(queue, mat_b_buf);
  QueueManager::getInstance().slow_full_sync();
  matmul.run(QueueManager::getInstance(), mat_a_buf.get(), mat_b_buf.get(), mat_res_buf.get());
  QueueManager::getInstance().slow_full_sync();*/
  BenchmarkApp app(argc, argv);
	
	app.run< Matmul >();
	QueueManager::sync();
}
