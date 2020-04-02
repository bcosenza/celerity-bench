#include "common.h"

#include <iostream>
namespace s = cl::sycl;
template <typename T> class VecAddKernel;

template <typename T>
class VecAddBench
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
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    output.resize(args.problem_size);

    for (size_t i =0; i < args.problem_size; i++) {
      input1[i] = static_cast<T>(i);
      input2[i] = static_cast<T>(i);
      output[i] = static_cast<T>(0);
    }

    input1_buf.initialize(input1.data(), s::range<1>(args.problem_size));
    input2_buf.initialize(input2.data(), s::range<1>(args.problem_size));
    output_buf.initialize(output.data(), s::range<1>(args.problem_size));
  }

  void run() {
  
    celerity::distr_queue& queue = QueueManager::getInstance();

    celerity::buffer<T,1> input1_buf(input1.data(), s::range<1>(args.problem_size));
    celerity::buffer<T,1> input2_buf(input2.data(), s::range<1>(args.problem_size));
    celerity::buffer<T,1> output_buf(output.data(), s::range<1>(args.problem_size));
  
    queue.submit([=](celerity::handler& cgh) {
      auto in1 = input1_buf.template get_access<s::access::mode::read>(cgh, celerity::access::one_to_one<1>());
      auto in2 = input2_buf.template get_access<s::access::mode::read>(cgh, celerity::access::one_to_one<1>());
      // Use discard_write here, otherwise the content of the host buffer must first be copied to device
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
      cl::sycl::range<1> ndrange {args.problem_size};

      cgh.parallel_for<class VecAddKernel<T>>(ndrange,
        [=](cl::sycl::id<1> gid) 
        {
          out[gid] = in1[gid] + in2[gid];
        });
    });

  }

  bool verify(VerificationSetting &ver) {
    //Triggers writeback
    //output_buf.reset();

    bool pass = true;
    for(size_t i=ver.begin[0]; i<ver.begin[0]+ver.range[0]; i++){
        auto expected = input1[i] + input2[i];
        if(expected != output[i]){
            pass = false;
            break;
        }
      }    
    return pass;
  }
  
  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "VectorAddition_";
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<VecAddBench<int>>();
  app.run<VecAddBench<long long>>();  
  app.run<VecAddBench<float>>();
  app.run<VecAddBench<double>>();
  return 0;
}
