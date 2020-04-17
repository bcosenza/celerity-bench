#include "common.h"

namespace s = cl::sycl;

class MicroBenchDRAMKernel;

class CopyBufferDummyKernel;

s::range<BENCH_DIMS> getBufferSize(size_t problemSize) {
#if BENCH_DIMS == 1
  return s::range<1>(problemSize * problemSize * problemSize / sizeof(BENCH_DATA_TYPE));
#elif BENCH_DIMS == 2
  return s::range<2>(problemSize * problemSize / sizeof(BENCH_DATA_TYPE), problemSize);
#elif BENCH_DIMS == 3
  return s::range<3>(problemSize / sizeof(BENCH_DATA_TYPE), problemSize, problemSize);
#endif
}

/**
 * Microbenchmark measuring DRAM bandwidth.
 */
class MicroBenchDRAM {
protected:
  BenchmarkArgs args;
  const s::range<BENCH_DIMS> buffer_size;
  // Since we cannot use explicit memory operations to initialize the input buffer,
  // we have to keep this around, unfortunately.
  std::vector<BENCH_DATA_TYPE> input;
  PrefetchedBuffer<BENCH_DATA_TYPE, BENCH_DIMS> input_buf;
  PrefetchedBuffer<BENCH_DATA_TYPE, BENCH_DIMS> output_buf;

public:
  MicroBenchDRAM(const BenchmarkArgs& args)
      : args(args), buffer_size(getBufferSize(args.problem_size)), input(buffer_size.size(), 33.f) {}

  void setup() {
    input_buf.initialize(input.data(), buffer_size);
    output_buf.initialize(buffer_size);
  }

  static ThroughputMetric getThroughputMetric(const BenchmarkArgs& args) {
    const double copiedGiB =
        getBufferSize(args.problem_size).size() * sizeof(BENCH_DATA_TYPE) / 1024.0 / 1024.0 / 1024.0;
    // Multiply by two as we are both reading and writing one element in each thread.
    return {copiedGiB * 2.0, "GiB"};
  }

  void run() {
    celerity::distr_queue& queue = QueueManager::getInstance();

    celerity::buffer<BENCH_DATA_TYPE, BENCH_DIMS>& a = input_buf.get();
    celerity::buffer<BENCH_DATA_TYPE, BENCH_DIMS>& b = output_buf.get();

    queue.submit([=](celerity::handler& cgh) {
      auto in = a.template get_access<s::access::mode::read>(cgh, celerity::access::one_to_one<BENCH_DIMS>());
      auto out = b.template get_access<s::access::mode::discard_write>(cgh, celerity::access::one_to_one<BENCH_DIMS>());
      // We spawn one work item for each buffer element to be copied.
      const s::range<BENCH_DIMS> global_size{buffer_size};
      cgh.parallel_for<MicroBenchDRAMKernel>(global_size, [=](s::id<BENCH_DIMS> gid) { out[gid] = in[gid]; });
    });
  }

  bool verify(VerificationSetting& ver) {
    bool pass = true;
    QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<s::access::mode::read>(cgh, s::range<BENCH_DIMS>(buffer_size));
      cgh.run([=,&pass]() {
        for(size_t i = 0; i < buffer_size[0]; ++i) {
          for(size_t j = 0; j < (BENCH_DIMS < 2 ? 1 : buffer_size[1]); ++j) {
            for(size_t k = 0; k < (BENCH_DIMS < 3 ? 1 : buffer_size[2]); ++k) {
#if BENCH_DIMS == 1
              if(result[i] != 33.f) {
                pass = false;
                break;
              }
#elif BENCH_DIMS == 2
              if(result[{i, j}] != 33.f) {
                pass = false;
                break;
              }
#elif BENCH_DIMS == 3
              if(result[{i, j, k}] != 33.f) {
                pass = false;
                break;
              }
#endif
            }
            if(!pass)
              break;
          }
          if(!pass)
            break;
        }
      });
    });
    QueueManager::sync();
    return pass;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_DRAM_";
    name << ReadableTypename<BENCH_DATA_TYPE>::name;
    name << "_" << BENCH_DIMS;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchDRAM>();

  return 0;
}
