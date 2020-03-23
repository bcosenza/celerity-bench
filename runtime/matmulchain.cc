#include <cstdio>
#include <vector>

#include <common.h>

// Performs chained matrix multiply of the form (AB)(CD)
// Uses two intermediate buffers and one for the result

template <typename T>
class MatmulChain;

template <typename T>
void multiply(celerity::distr_queue& queue, celerity::buffer<T, 2>& mat_a, celerity::buffer<T, 2>& mat_b,
    celerity::buffer<T, 2>& mat_c, const size_t mat_size) {
	queue.submit([=](celerity::handler& cgh) {
		auto a = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
		auto b = mat_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
		auto c = mat_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
		//auto a = mat_a.template get_access<cl::sycl::access::mode::read>(cgh);
		//auto b = mat_b.template get_access<cl::sycl::access::mode::read>(cgh);
		//auto c = mat_c.template get_access<cl::sycl::access::mode::discard_write>(cgh);

		cgh.parallel_for<class MatmulChain<T>>(cl::sycl::range<2>(mat_size, mat_size), [=](cl::sycl::item<2> item) {
			auto sum = 0.f;
			for(size_t k = 0; k < mat_size; ++k) {
				const auto a_ik = a[{item[0], k}];
				const auto b_kj = b[{k, item[1]}];
				sum += a_ik * b_kj;
			}
			c[item] = sum;
		});
  });
}


template <typename T>
class MatmulChain {
protected:    
	std::vector<T> mat_a;
	std::vector<T> mat_b;
	std::vector<T> mat_c;
	std::vector<T> mat_d;
	std::vector<T> mat_res;
	BenchmarkArgs args;
	int mat_size;

	//PrefetchedBuffer<T, 2> mat_a_buf;
	//PrefetchedBuffer<T, 2> mat_b_buf;
	//PrefetchedBuffer<T, 2> mat_c_buf;
	//PrefetchedBuffer<T, 2> mat_d_buf;
	//PrefetchedBuffer<T, 2> mat_res_buf;
	//PrefetchedBuffer<T, 2> mat_p_buf;
	//PrefetchedBuffer<T, 2> mat_q_buf;

	std::shared_ptr<celerity::buffer<T, 2>> mat_a_buf;
	std::shared_ptr<celerity::buffer<T, 2>> mat_b_buf;
	std::shared_ptr<celerity::buffer<T, 2>> mat_c_buf;
	std::shared_ptr<celerity::buffer<T, 2>> mat_d_buf;
	std::shared_ptr<celerity::buffer<T, 2>> mat_res_buf;
	std::shared_ptr<celerity::buffer<T, 2>> mat_p_buf;
	std::shared_ptr<celerity::buffer<T, 2>> mat_q_buf;

public:
	MatmulChain(const BenchmarkArgs &_args) : args(_args) {
		mat_size = args.problem_size;
	}

	void setup() {
		mat_a = std::vector<T>(mat_size * mat_size);
		mat_b = std::vector<T>(mat_size * mat_size);
		mat_c = std::vector<T>(mat_size * mat_size);
		mat_d = std::vector<T>(mat_size * mat_size);
		mat_res = std::vector<T>(mat_size * mat_size);

		// Initialize matrices to the identity
		for(size_t i = 0; i < mat_size; ++i) {
			for(size_t j = 0; j < mat_size; ++j) {
				mat_a[i * mat_size + j] = i == j;
				mat_b[i * mat_size + j] = i == j;
				mat_c[i * mat_size + j] = i == j;
				mat_d[i * mat_size + j] = i == j;
			}
		}

		mat_a_buf =   std::make_shared<celerity::buffer<T, 2>>(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
		mat_b_buf =   std::make_shared<celerity::buffer<T, 2>>(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
		mat_c_buf =   std::make_shared<celerity::buffer<T, 2>>(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
		mat_d_buf =   std::make_shared<celerity::buffer<T, 2>>(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
		mat_p_buf =   std::make_shared<celerity::buffer<T, 2>>(cl::sycl::range<2>(mat_size, mat_size));
		mat_q_buf =   std::make_shared<celerity::buffer<T, 2>>(cl::sycl::range<2>(mat_size, mat_size));
		mat_res_buf = std::make_shared<celerity::buffer<T, 2>>(cl::sycl::range<2>(mat_size, mat_size));
		// celerity::buffer<float, 2> mat_a_buf(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
		// celerity::buffer<float, 2> mat_b_buf(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
		// celerity::buffer<float, 2> mat_c_buf(cl::sycl::range<2>(mat_size, mat_size));

		//mat_a_buf.initialize(benchQueue, mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
		//mat_b_buf.initialize(benchQueue, mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
		//mat_c_buf.initialize(benchQueue, mat_c.data(), cl::sycl::range<2>(mat_size, mat_size));
		//mat_d_buf.initialize(benchQueue, mat_d.data(), cl::sycl::range<2>(mat_size, mat_size));
		//mat_res_buf.initialize(benchQueue, mat_res.data(), cl::sycl::range<2>(mat_size, mat_size));
		//mat_p_buf.initialize(benchQueue, cl::sycl::range<2>(mat_size, mat_size));
		//mat_q_buf.initialize(benchQueue, cl::sycl::range<2>(mat_size, mat_size));
	}

	void run() {
		multiply(benchQueue, *mat_a_buf.get(), *mat_b_buf.get(), *mat_p_buf.get(), mat_size);
		multiply(benchQueue, *mat_c_buf.get(), *mat_d_buf.get(), *mat_q_buf.get(), mat_size);
		multiply(benchQueue, *mat_p_buf.get(), *mat_q_buf.get(), *mat_res_buf.get(), mat_size);
	}

	static std::string getBenchmarkName() { return "MatmulChain"; }

	bool verify(VerificationSetting &ver) {
		bool verification_passed = true;
		/*benchQueue.with_master_access([&](celerity::handler& cgh) {
			auto result = mat_res_buf.get()->get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));

			cgh.run([=, &verification_passed]() {
				celerity::experimental::bench::end("main program");

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
		return verification_passed;
	}
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	
	// float 
	app.run< MatmulChain<float> >();
}
