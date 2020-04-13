#include "common.h"

#include <iostream>
namespace s = cl::sycl;

//const int neigh_size = 1;

const int fixed_size = 2;

template <typename T> class OneToOneMapperKernel;
template <typename T> class NeighborhoodMapperKernel;
template <typename T> class SliceMapperKernel;
template <typename T> class FixedMapperKernel;
template <typename T> class AllMapperKernel;

template <typename T>
class RangeMappersBench
{
protected:    
  std::vector<T> input1;
  std::vector<T> input2;
  std::vector<T> output;
  BenchmarkArgs args;
  size_t neigh_size;

 PrefetchedBuffer<T, 2> input1_buf;
 PrefetchedBuffer<T, 2> input2_buf;
 PrefetchedBuffer<T, 2> output_buf;

public:
  RangeMappersBench(const BenchmarkArgs &_args, size_t _neigh_size) : args(_args), neigh_size(_neigh_size) {}
  
  void setup() {
    // host memory intilization
    input1.resize(args.problem_size*args.problem_size);
    input2.resize(args.problem_size*args.problem_size);
    output.resize(args.problem_size*args.problem_size);

    for (size_t i = 0; i < args.problem_size*args.problem_size; i++) {
      input1[i] = static_cast<T>(i);
      input2[i] = static_cast<T>(i);
      output[i] = static_cast<T>(0);
    }

    input1_buf.initialize(input1.data(), s::range<2>(args.problem_size, args.problem_size));
    input2_buf.initialize(input2.data(), s::range<2>(args.problem_size, args.problem_size));
    output_buf.initialize(output.data(), s::range<2>(args.problem_size, args.problem_size));
  }

  void one_to_one(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class OneToOneMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

  void slice(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::slice<2>(0));

      cgh.parallel_for<class SliceMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

  void neighborhood(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 2>& buf_c, size_t neigh_size) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));

      cgh.parallel_for<class NeighborhoodMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }


  void fixed(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::fixed<2>({{fixed_size, fixed_size},{fixed_size, fixed_size}}));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::fixed<2>({{fixed_size, fixed_size},{fixed_size, fixed_size}}));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::fixed<2>({{fixed_size, fixed_size},{fixed_size, fixed_size}}));

      cgh.parallel_for<class FixedMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

  void all(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::all<2>());

      cgh.parallel_for<class AllMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }         

  void run() {
  
    // Matrix addition using one_to_one ranage mapper
    one_to_one(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using neighbourhood ranage mapper
    neighborhood(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get(), neigh_size);

    // Matrix addition using slice ranage mapper
    slice(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using fixed ranage mapper
    fixed(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using all ranage mapper
    all(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

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
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  size_t neigh_size = 1;

  app.run<RangeMappersBench<int>>(neigh_size);
  return 0;
}
