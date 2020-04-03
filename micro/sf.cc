#include "common.h"

namespace s = cl::sycl;

template <typename DataT, int N>
class MicroBenchSpecialFuncKernel;

/**
 * Microbenchmark stressing the special function units.
 */
template <typename DataT, int Iterations = 16>
class MicroBenchSpecialFunc {
protected:
  std::vector<DataT> input;
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> input_buf;
  PrefetchedBuffer<DataT, 1> output_buf;

public:
  MicroBenchSpecialFunc(const BenchmarkArgs& args) : args(args) {}

  void setup() {
    input.resize(args.problem_size, DataT{3.14});

    input_buf.initialize(input.data(), s::range<1>(args.problem_size));
    output_buf.initialize(s::range<1>(args.problem_size));
  }

  static ThroughputMetric getThroughputMetric(const BenchmarkArgs& args) {
    const double OP = args.problem_size * Iterations * 3;
    return {OP / 1024.0 / 1024.0 / 1024.0, "GOP"};
  }

  void run() {
    celerity::distr_queue& queue = QueueManager::getInstance();

    celerity::buffer<DataT, 1>& a = input_buf.get();
    celerity::buffer<DataT, 1>& b = output_buf.get();

    queue.submit([=](celerity::handler& cgh) {
      auto in = a.template get_access<s::access::mode::read>(cgh, celerity::access::one_to_one<1>());
      auto out = b.template get_access<s::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());

      cgh.parallel_for<MicroBenchSpecialFuncKernel<DataT, Iterations>>(
          s::range<1>{args.problem_size}, [=](s::id<1> gid) {
            DataT v0, v1, v2;
            v0 = in[gid];
            v1 = v2 = v0;
            for(int i = 0; i < Iterations; ++i) {
              v0 = s::cos(v1);
              v1 = s::sin(v2);
              v2 = s::tan(v0);
            }
            out[gid] = v2;
          });
    });
  }

  bool verify(VerificationSetting& ver) {
    DataT v0, v1, v2;
    v0 = DataT{3.14};
    v1 = v2 = v0;
    for(int i = 0; i < Iterations; ++i) {
      v0 = s::cos(v1);
      v1 = s::sin(v2);
      v2 = s::tan(v0);
    }
    const DataT expected = v2;
    bool pass = true;

    QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<s::access::mode::read>(cgh, cl::sycl::range<1>(args.problem_size));
        cgh.run([=,&pass]() {
        for(size_t i = 0; i < args.problem_size; ++i) {
          constexpr DataT EPSILON = 1e-5;
          if(std::abs(result[i] - expected) > EPSILON) {
            pass = false;
            break;
          }
        }
      });
    });
    return pass;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_sf_";
    name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchSpecialFunc<float>>();
  app.run<MicroBenchSpecialFunc<double>>();

  return 0;
}
