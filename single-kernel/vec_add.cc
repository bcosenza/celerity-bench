#include "common.h"

#include <iostream>
namespace s = cl::sycl;
class VecAddKernel;

class VecAddBench
{
protected:    
  std::vector<BENCH_DATA_TYPE> input1;
  std::vector<BENCH_DATA_TYPE> input2;
  std::vector<BENCH_DATA_TYPE> output;
  BenchmarkArgs args;

 PrefetchedBuffer<BENCH_DATA_TYPE, 1> input1_buf;
 PrefetchedBuffer<BENCH_DATA_TYPE, 1> input2_buf;
 PrefetchedBuffer<BENCH_DATA_TYPE, 1> output_buf;

public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {
    // host memory intilization
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    output.resize(args.problem_size);

    for (size_t i = 0; i < args.problem_size; i++) {
      input1[i] = static_cast<BENCH_DATA_TYPE>(i);
      input2[i] = static_cast<BENCH_DATA_TYPE>(i);
      output[i] = static_cast<BENCH_DATA_TYPE>(0);
     // std::cout << input1[i] << ":" << input2[i] << std::endl;
    }

    input1_buf.initialize(input1.data(), s::range<1>(args.problem_size));
    input2_buf.initialize(input2.data(), s::range<1>(args.problem_size));
    output_buf.initialize(output.data(), s::range<1>(args.problem_size));
  }

  void run() {
  
    celerity::distr_queue& queue = QueueManager::getInstance();

    celerity::buffer<BENCH_DATA_TYPE,1>& a = input1_buf.get();
    celerity::buffer<BENCH_DATA_TYPE,1>& b = input2_buf.get();
    celerity::buffer<BENCH_DATA_TYPE,1>& c = output_buf.get();

    queue.submit([=](celerity::handler& cgh) {
      auto in1 = a.template get_access<s::access::mode::read>(cgh, celerity::access::one_to_one<1>());
      auto in2 = b.template get_access<s::access::mode::read>(cgh, celerity::access::one_to_one<1>());
      // Use discard_write here, otherwise the content of the host buffer must first be copied to device
      auto out = c.template get_access<s::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
      cl::sycl::range<1> ndrange {args.problem_size};

      cgh.parallel_for<class VecAddKernel>(ndrange,
        [=](cl::sycl::id<1> gid) 
        {
          out[gid] = in1[gid] + in2[gid];
        });
    });

  }

  bool verify(VerificationSetting &ver) {
    bool verification_passed = true;
    QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(args.problem_size));
      cgh.run([=, &verification_passed]() {
      for(size_t i = 0; i < args.problem_size; i++){
          auto expected = input1[i] + input2[i];
          if(expected != result[i]){
              verification_passed = false;
              break;
          }
        }  
      });
    });
    QueueManager::sync();
    return verification_passed;
  }
  
  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "VectorAddition_";
    name << ReadableTypename<BENCH_DATA_TYPE>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<VecAddBench>();
  return 0;
}
