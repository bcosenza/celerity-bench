#include "common.h"

#include <iostream>
namespace s = cl::sycl;
template <typename T> class RangeMapperKernel;

template <typename T>
class RangeMappersBench
{
protected:    
  std::vector<T> input1;
  std::vector<T> input2;
  std::vector<T> output;
  BenchmarkArgs args;

 PrefetchedBuffer<T, 1> input1_buf;
 PrefetchedBuffer<T, 1> input2_buf;
 PrefetchedBuffer<T, 1> output_buf;

public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {}
  
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
    input2_buf.initialize(input2.data(), s::range<1>(args.problem_size, args.problem_size));
    output_buf.initialize(output.data(), s::range<1>(args.problem_size, args.problem_size));
  }

  void one_to_one(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class OneToOneMapper<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        for(size_t k = 0; k < args.problem_size; ++k) {
          c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
        }
      });
    });
  } 

  void run() {
  
    celerity::distr_queue& queue = QueueManager::getInstance();

    // Matrix addition using one_to_one ranage mapper
    one_to_one(queue, input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using neighbourhood ranage mapper
   // neighbourhood(queue, input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using slice ranage mapper
   // slice(queue, input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using fixed ranage mapper
    fixed(queue, input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using all ranage mapper
   // all(queue, input1_buf.get(), input2_buf.get(), output_buf.get());

  }

  bool verify(VerificationSetting &ver) {
    bool verification_passed = true;
    QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(args.problem_size, args.problem_size));
      cgh.run([=, &verification_passed]() {
        for(size_t i = 0; i < args.problem_size; i++){
          for (size_t j = 0; j < args.problem_size; j++) {
            auto expected = input1[i][j] + input2[i][j];
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
  app.run<RangeMappersBench<int>>();
  return 0;
}
