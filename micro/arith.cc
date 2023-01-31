#include "common.h"

namespace s = cl::sycl;

template <int Iterations>
class MicroBenchArithmeticKernel;

/**
 * Microbenchmark stressing the main arithmetic units.
 */
template <int Iterations = 512>
class MicroBenchArithmetic {
protected:
  std::vector<BENCH_DATA_TYPE> input;
  BenchmarkArgs args;

  PrefetchedBuffer<BENCH_DATA_TYPE, 1> input_buf;
  PrefetchedBuffer<BENCH_DATA_TYPE, 1> output_buf;

public:
  MicroBenchArithmetic(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    input.resize(args.problem_size, BENCH_DATA_TYPE{1});

    input_buf.initialize(input.data(), s::range<1>(args.problem_size));
    output_buf.initialize(s::range<1>(args.problem_size));
  }

  static ThroughputMetric getThroughputMetric(const BenchmarkArgs& args) {
    if constexpr(std::is_same_v<BENCH_DATA_TYPE, float>) {
      // Multiply everything times two as we are doing FMAs.
      const double FLOP = args.problem_size * Iterations * 2 * 2;
      return {FLOP / 1024.0 / 1024.0 / 1024.0, "SP GFLOP"};
    }
    if constexpr(std::is_same_v<BENCH_DATA_TYPE, double>) {
      // Multiply everything times two as we are doing FMAs.
      const double DFLOP = args.problem_size * Iterations * 2 * 2;
      return {DFLOP / 1024.0 / 1024.0 / 1024.0, "DP GFLOP"};
    }
    if constexpr(std::is_same_v<BENCH_DATA_TYPE, int>) {
      // Multiply everything times two as we are doing MAD.
      const double OP = args.problem_size * Iterations * 2 * 2;
      return {OP / 1024.0 / 1024.0 / 1024.0, "GOP"};
    }
    return {};
  }

  void run() {
    celerity::distr_queue& queue = QueueManager::getInstance();

    celerity::buffer<BENCH_DATA_TYPE, 1>& a = input_buf.get();
    celerity::buffer<BENCH_DATA_TYPE, 1>& b = output_buf.get();

    queue.submit([=](celerity::handler& cgh) {
      auto in = a.template get_access<s::access::mode::read>(cgh, celerity::access::one_to_one<1>());
      auto out = b.template get_access<s::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());

      cgh.parallel_for<MicroBenchArithmeticKernel<Iterations>>(
          s::range<1>{args.problem_size}, [=](cl::sycl::id<1> gid) {
            BENCH_DATA_TYPE a1 = in[gid];
            const BENCH_DATA_TYPE a2 = a1;

            for(int i = 0; i < Iterations; ++i) {
              // We do two operations to ensure the value remains 1 and doesn't grow indefinitely.
              a1 = a1 * a1 + a1;
              a1 = a1 * a2 - a2;
            }

            out[gid] = a1;
          });
    });
  }

  bool verify(VerificationSetting& ver) {
    bool pass = true;
    QueueManager::getInstance().submit(celerity::allow_by_ref, [&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<s::access::mode::read>(cgh, cl::sycl::range<1>(args.problem_size));
      cgh.host_task(celerity::on_master_node, [=, &pass](celerity::partition<2> part) {

			auto result = mat_a_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::one_to_one<2>());

			cgh.host_task(range, [=, &verification_passed](celerity::partition<2> part) {
    QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<s::access::mode::read>(cgh, cl::sycl::range<1>(args.problem_size));
      cgh.run([=,&pass]() {
        for(size_t i = 0; i < args.problem_size; ++i) {
          if(result[i] != BENCH_DATA_TYPE{1}) {
            pass = false;
            break;
          }
        }
      });
    });
    QueueManager::sync();
    return pass;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_Arith_";
    name << ReadableTypename<BENCH_DATA_TYPE>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchArithmetic<BENCH_COMP_ITERS>>();

  // app.run<MicroBenchArithmetic<int>>();
  // app.run<MicroBenchArithmetic<float>>();
  // app.run<MicroBenchArithmetic<double>>();

  return 0;
}
