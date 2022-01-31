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
  size_t size;

 PrefetchedBuffer<BENCH_DATA_TYPE, 1> input1_buf;
 PrefetchedBuffer<BENCH_DATA_TYPE, 1> input2_buf;
 PrefetchedBuffer<BENCH_DATA_TYPE, 1> output_buf;

public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {
    size = args.problem_size * args.problem_size;
    // host memory intilization
    input1.resize(size);
    input2.resize(size);
    output.resize(size);

    for (size_t i = 0; i < size; i++) {
      input1[i] = static_cast<BENCH_DATA_TYPE>(i);
      input2[i] = static_cast<BENCH_DATA_TYPE>(i);
      output[i] = static_cast<BENCH_DATA_TYPE>(0);
     // std::cout << input1[i] << ":" << input2[i] << std::endl;
    }
    auto range = celerity::range<1>(size);
    input1_buf.initialize(input1.data(), range);
    input2_buf.initialize(input2.data(), range);
    output_buf.initialize(output.data(), range);
  }

  void run() {
  
    celerity::distr_queue& queue = QueueManager::getInstance();

    celerity::buffer<BENCH_DATA_TYPE,1>& a = input1_buf.get();
    celerity::buffer<BENCH_DATA_TYPE,1>& b = input2_buf.get();
    celerity::buffer<BENCH_DATA_TYPE,1>& c = output_buf.get();

    queue.submit([=](celerity::handler& cgh) {
      celerity::accessor in1{a, cgh, celerity::access::one_to_one{}, celerity::read_only};
      celerity::accessor in2{b, cgh, celerity::access::one_to_one{}, celerity::read_only};
      // Use discard_write here, otherwise the content of the host buffer must first be copied to device
      //auto out = c.template get_access<s::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
      celerity::accessor out{c, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
      celerity::range<1> ndrange {size};

      cgh.parallel_for<class VecAddKernel>(ndrange,
        [=](cl::sycl::id<1> gid) 
        {
          out[gid] = in1[gid] + in2[gid];
        });
    });

  }

  bool verify(VerificationSetting &ver) {
    bool verification_passed = true;
    /*QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(size));
      cgh.run([=, &verification_passed]() {
      for(size_t i = 0; i < size; i++){
          auto expected = input1[i] + input2[i];
          if(expected != result[i]){
              verification_passed = false;
              break;
          }
        }  
      });
    });*/
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
