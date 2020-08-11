#include "common.h"

#include <iostream>
namespace s = cl::sycl;

//const int neigh_size = 1;

const int fixed_size = 2;

class OneToOneMapperKernel;
class NeighborhoodMapperKernel;
class SliceMapperKernelX;
class FixedMapperKernelY;
class AllMapperKernel;

class RangeMappersBench
{
protected:    
  std::vector<BENCH_DATA_TYPE> input1;
  std::vector<BENCH_DATA_TYPE> input2;
  std::vector<BENCH_DATA_TYPE> output;
  BenchmarkArgs args;
  size_t neigh_size_limit;
  size_t fixed_size_limit;

 PrefetchedBuffer<BENCH_DATA_TYPE, 2> input1_buf;
 PrefetchedBuffer<BENCH_DATA_TYPE, 2> input2_buf;
 PrefetchedBuffer<BENCH_DATA_TYPE, 2> output_buf;

public:
  RangeMappersBench(const BenchmarkArgs &_args, size_t _neigh_size_limit, size_t _fixed_size_limit) : args(_args), neigh_size_limit(_neigh_size_limit), fixed_size_limit(_fixed_size_limit) {}
  
  void setup() {
    // host memory intilization
    input1.resize(args.problem_size*args.problem_size);
    input2.resize(args.problem_size*args.problem_size);
    output.resize(args.problem_size*args.problem_size);

    for (size_t i = 0; i < args.problem_size*args.problem_size; i++) {
      input1[i] = static_cast<BENCH_DATA_TYPE>(i);
      input2[i] = static_cast<BENCH_DATA_TYPE>(i);
      output[i] = static_cast<BENCH_DATA_TYPE>(0);
    }

    input1_buf.initialize(input1.data(), s::range<2>(args.problem_size, args.problem_size));
    input2_buf.initialize(input2.data(), s::range<2>(args.problem_size, args.problem_size));
    output_buf.initialize(output.data(), s::range<2>(args.problem_size, args.problem_size));
  }

#if defined(BENCH_MAPPER_ONE_TO_ONE)
  void one_to_one(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class OneToOneMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }
#elif defined(BENCH_MAPPER_SLICE_X)
  void slicex(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::slice<2>(0));

      cgh.parallel_for<class SliceMapperKernelX>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }
#elif defined(BENCH_MAPPER_SLICE_Y)
  void slicey(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::slice<2>(1));

      cgh.parallel_for<class SliceMapperKernelY>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

#elif defined(BENCH_MAPPER_NEIGHBOURHOOD)
  void neighborhood(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c, size_t neigh_size) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));

      cgh.parallel_for<class NeighborhoodMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

#elif defined(BENCH_MAPPER_FIXED)
  void fixed(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c, size_t fixed_size) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::fixed<2>({{fixed_size, fixed_size},{fixed_size, fixed_size}}));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::fixed<2>({{fixed_size, fixed_size},{fixed_size, fixed_size}}));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::fixed<2>({{fixed_size, fixed_size},{fixed_size, fixed_size}}));

      cgh.parallel_for<class FixedMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

#elif defined(BENCH_MAPPER_ALL)
  void all(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::all<2>());

      cgh.parallel_for<class AllMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }
#endif

  void run() {
#if defined(BENCH_MAPPER_ONE_TO_ONE)
    // Matrix addition using one_to_one range mapper
    one_to_one(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());
#elif defined(BENCH_MAPPER_NEIGHBOURHOOD)
    // Matrix addition using neighbourhood range mapper
    for (size_t neigh_size = 1; neigh_size < neigh_size_limit; neigh_size++)
      neighborhood(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get(), neigh_size);
#elif defined(BENCH_MAPPER_SLICE_X)
    // Matrix addition using slice range mapper
    slicex(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());
#elif defined(BENCH_MAPPER_SLICE_Y)
    // Matrix addition using slice range mapper
    slicey(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());    
#elif defined(BENCH_MAPPER_FIXED)
    // Matrix addition using fixed range mapper
    for (size_t fixed_size = 1; fixed_size < fixed_size_limit; fixed_size++)
      fixed(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get(), fixed_size);
#elif defined(BENCH_MAPPER_ALL)
    // Matrix addition using all range mapper
    all(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());
#endif
  }

  bool verify(VerificationSetting &ver) {
    bool verification_passed = true;
    QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(args.problem_size, args.problem_size));
      cgh.run([=, &verification_passed]() {
        for(size_t i = 0; i < args.problem_size; i++){
          for (size_t j = 0; j < args.problem_size; j++) {
            auto expected = input1[i*args.problem_size + j] + input2[i*args.problem_size + j];
            if(expected != result[i][j]){
                verification_passed = false;
                break;
            }
          } 
        } 
      });
    });
    QueueManager::sync();
    return verification_passed;
  }  
  
  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "RangeMappers_";
    name << ReadableTypename<BENCH_DATA_TYPE>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  size_t neigh_size_limit = 16;
  size_t fixed_size_limit = 4;

  app.run<RangeMappersBench>(neigh_size_limit, fixed_size_limit);
  return 0;
}
