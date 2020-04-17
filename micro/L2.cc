#include "common.h"

#include <iostream>

namespace s = cl::sycl;

class MicroBenchL2Kernel;

/* Microbenchmark stressing the main arithmetic units. */
class MicroBenchL2
{
protected:
    std::vector<BENCH_DATA_TYPE> input;
    BenchmarkArgs args;

    PrefetchedBuffer<BENCH_DATA_TYPE, 1> input_buf;
    PrefetchedBuffer<BENCH_DATA_TYPE, 1> output_buf;
public:
  MicroBenchL2(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
    // buffers initialized to a default value 
    input. resize(args.problem_size, 10);

    input_buf.initialize(input.data(), s::range<1>(args.problem_size));
    output_buf.initialize(s::range<1>(args.problem_size));
  }

  void run(){
    celerity::distr_queue& queue = QueueManager::getInstance();

    celerity::buffer<BENCH_DATA_TYPE, 1>& a = input_buf.get();
    celerity::buffer<BENCH_DATA_TYPE, 1>& b = output_buf.get();

    queue.submit([=](celerity::handler& cgh) {
      auto in  = a.template get_access<s::access::mode::read>(cgh, celerity::access::one_to_one<1>());
      auto out = b.template get_access<s::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
      cl::sycl::range<1> ndrange {args.problem_size};

      cgh.parallel_for<MicroBenchL2Kernel>(ndrange,
        [=](cl::sycl::id<1> gid)
      {
        BENCH_DATA_TYPE r0;
        for (int i=0;i<BENCH_COMP_ITERS;i++) {
            r0 = in[gid];
            out[gid] = r0; 
        }
        out[gid] = r0;
      });
    }); // submit
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_L2_";
    name << ReadableTypename<BENCH_DATA_TYPE>::name << "_";
    name << BENCH_COMP_ITERS;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);

  app.run< MicroBenchL2 >();

  return 0;
}



