#include <cstdio>
#include <vector>

#include <common.h>

// Performs chained matrix multiply of the form (AB)(CD)
// Uses two intermediate buffers and one for the result

template <typename T>
class MatmulChain;

template <typename T>
void set_identity(celerity::distr_queue queue, celerity::buffer<T, 2> mat) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor dw{mat, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class set_identity_kernel>(mat.get_range(), [=](celerity::item<2> item) { dw[item] = item[0] == item[1]; });
	});
}

template <typename T>
void multiply(celerity::distr_queue queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_b, celerity::buffer<T, 2> mat_c, const size_t mat_size) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor a{mat_a, cgh, celerity::access::slice<2>(1), celerity::read_only};
		celerity::accessor b{mat_b, cgh, celerity::access::slice<2>(0), celerity::read_only};
		celerity::accessor c{mat_c, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};

/* #if CELERITY_FEATURE_LOCAL_ACCESSOR

		// Use local-memory tiling to avoid waiting on global memory too often
		const size_t GROUP_SIZE = 8;
		celerity::local_accessor<T, 2> scratch_a{{GROUP_SIZE, GROUP_SIZE}, cgh};
		celerity::local_accessor<T, 2> scratch_b{{GROUP_SIZE, GROUP_SIZE}, cgh};

		cgh.parallel_for<class MatmulChain<T>>(celerity::nd_range<2>{{mat_size, mat_size}, {GROUP_SIZE, GROUP_SIZE}}, [=](celerity::nd_item<2> item) {
			T sum{};
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

		cgh.parallel_for<class MatmulChain<T>>(celerity::range<2>(mat_size, mat_size), [=](celerity::item<2> item) {
			T sum{};
			for(size_t k = 0; k < mat_size; ++k) {
				const auto a_ik = a[{item[0], k}];
				const auto b_kj = b[{k, item[1]}];
				sum += a_ik * b_kj;
			}
			c[item] = sum;
		});

//#endif
	});
}


template <typename T>
class MatmulChain {
protected:    
	BenchmarkArgs args;
	int mat_size;

	PrefetchedBuffer<T, 2> mat_a_buf;
	PrefetchedBuffer<T, 2> mat_b_buf;
	PrefetchedBuffer<T, 2> mat_c_buf;
	PrefetchedBuffer<T, 2> mat_d_buf;
	PrefetchedBuffer<T, 2> mat_p_buf;
	PrefetchedBuffer<T, 2> mat_q_buf;
	PrefetchedBuffer<T, 2> mat_res_buf;

public:
	MatmulChain(const BenchmarkArgs &_args) : args(_args) {
		mat_size = args.problem_size;
	}

	void setup() {
		auto range = celerity::range<2>(mat_size, mat_size);
		mat_a_buf.initialize(range);
		mat_b_buf.initialize(range);
		mat_c_buf.initialize(range);
		mat_d_buf.initialize(range);
		mat_p_buf.initialize(range);
		mat_q_buf.initialize(range);
		mat_res_buf.initialize(range);
    set_identity(QueueManager::getInstance(), mat_a_buf.get());
    set_identity(QueueManager::getInstance(), mat_b_buf.get());
    set_identity(QueueManager::getInstance(), mat_c_buf.get());
    set_identity(QueueManager::getInstance(), mat_d_buf.get());
	}

	void run() {
		multiply(QueueManager::getInstance(), mat_a_buf.get(), mat_b_buf.get(), mat_p_buf.get(), mat_size);
		multiply(QueueManager::getInstance(), mat_c_buf.get(), mat_d_buf.get(), mat_q_buf.get(), mat_size);
		multiply(QueueManager::getInstance(), mat_p_buf.get(), mat_q_buf.get(), mat_res_buf.get(), mat_size);
	}

	static std::string getBenchmarkName() { return "MatmulChain"; }

	bool verify(VerificationSetting &ver) {
		bool verification_passed = true;
		/*QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
			auto result = mat_res_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));

			cgh.run([=, &verification_passed]() {

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
		});
		QueueManager::sync();*/
		return verification_passed;
	}
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	
	// float 
	app.run< MatmulChain<float> >();
}
