#include "common.h"

#include <iostream>
namespace s = cl::sycl;

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
  RangeMappersBench(const BenchmarkArgs &_args) : args(_args) {}
  
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

#if defined(BENCH_MAPPER_ONE_TO_ONE_ALL) || defined(BENCH_MAPPER_ALL_ONE_TO_ONE)
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

  void all(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class AllMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }


#elif defined(BENCH_MAPPER_ONE_TO_ONE_NEIGHBOURHOOD) 
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

  void neighborhood(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c, size_t neigh_size) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class NeighborhoodMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

#elif defined(BENCH_MAPPER_ONE_TO_ONE_SLICEX) 
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

  void slicex(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class SliceMapperKernelX>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

#elif defined(BENCH_MAPPER_ONE_TO_ONE_SLICEY) 
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

  void slicey(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class SliceMapperKernelY>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

#elif defined(BENCH_MAPPER_ONE_TO_ONE_ONE_TO_ONE) 
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

#elif defined(BENCH_MAPPER_ALL_ALL) 
  void all(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class AllMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

#elif defined(BENCH_MAPPER_NEIGHBOURHOOD_NEIGHBOURHOOD) 
 void neighborhood(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c, size_t neigh_size) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class NeighborhoodMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  } 

#elif defined(BENCH_MAPPER_ALL_NEIGHBOURHOOD) 
  void all(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class AllMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

  void neighborhood(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c, size_t neigh_size) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(neigh_size, neigh_size));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class NeighborhoodMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

#elif defined(BENCH_MAPPER_ALL_SLICEX) 
  void all(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class AllMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

  void slicex(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class SliceMapperKernelX>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  } 

#elif defined(BENCH_MAPPER_ALL_SLICEY) 
  void all(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class AllMapperKernel>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

  void slicey(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class SliceMapperKernelX>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  } 

#elif defined(BENCH_MAPPER_SLICEY_SLICEX) || defined(BENCH_MAPPER_SLICEX_SLICEY)

  void slicey(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class SliceMapperKernelX>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

  void slicex(celerity::distr_queue& queue, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_a, celerity::buffer<BENCH_DATA_TYPE, 2>& buf_b,celerity::buffer<BENCH_DATA_TYPE, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class SliceMapperKernelX>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

#endif

  void run() {

    #if defined(BENCH_MAPPER_ONE_TO_ONE_ALL)
    // Matrix addition using one_to_one range mapper
    // c = a+b
    one_to_one(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using all range mapper
    // a = b+c
    all(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get());
    
    #elif defined(BENCH_MAPPER_ALL_ONE_TO_ONE)
    // Matrix addition using one_to_one range mapper
    // c = a+b
    all(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using all range mapper
    // a = b+c
    one_to_one(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get());    
    
    #elif defined(BENCH_MAPPER_ONE_TO_ONE_NEIGHBOURHOOD)
    // Matrix addition using one_to_one range mapper
    // c = a+b
    one_to_one(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using neighbourhood range mapper
    // a = b+c
    neighborhood(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get(), 3);
    
    #elif defined(BENCH_MAPPER_ONE_TO_ONE_SLICEX)
    // Matrix addition using one_to_one range mapper
    // c = a+b
    one_to_one(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using slice range mapper
    // a = b+c
    slicex(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get());
   
    #elif defined(BENCH_MAPPER_ONE_TO_ONE_SLICEY)
    // Matrix addition using one_to_one range mapper
    // c = a+b
    one_to_one(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using slice range mapper
    // a = b+c
    slicey(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get());

    #elif defined(BENCH_MAPPER_ONE_TO_ONE_ONE_TO_ONE)
    // Matrix addition using one_to_one range mapper
    // c = a+b
    one_to_one(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using one_to_one range mapper
    // a = b+c
    one_to_one(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get());
   
    #elif defined(BENCH_MAPPER_ALL_ALL)
    // Matrix addition using all range mapper
    // c = a+b
    all(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using all range mapper
    // a = b+c
    all(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get());

    #elif defined(BENCH_MAPPER_NEIGHBOURHOOD_NEIGHBOURHOOD)
    // Matrix addition using neighbourhood range mapper
    // c = a+b
    neighborhood(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get(), 3);

    // Matrix addition using neighbourhood range mapper
    // a = b+c
    neighborhood(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get(), 3);

    #elif defined(BENCH_MAPPER_ALL_NEIGHBOURHOOD)
    // Matrix addition using all range mapper
    // c = a+b
    all(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using neighbourhood range mapper
    // a = b+c
    neighborhood(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get(), 3);

    #elif defined(BENCH_MAPPER_SLICEX_SLILCEY)
    // Matrix addition using all range mapper
    // c = a+b
    slicex(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using all range mapper
    // a = b+c
    slicey(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get());

    #elif defined(BENCH_MAPPER_SLICEY_SLILCEX)
    // Matrix addition using all range mapper
    // c = a+b
    slicey(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using all range mapper
    // a = b+c
    slicex(QueueManager::getInstance(), input2_buf.get(), output_buf.get(), input1_buf.get());
    #endif           

  // QueueManager::sync();  
  
  }

  bool verify(VerificationSetting &ver) {
    bool verification_passed = true;
    QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = input1_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(args.problem_size, args.problem_size));
      cgh.run([=, &verification_passed]() {
        for(size_t i = 0; i < args.problem_size; i++){
          for (size_t j = 0; j < args.problem_size; j++) {
            auto temp = input1[i*args.problem_size + j] + input2[i*args.problem_size + j];
            auto expected = input2[i*args.problem_size + j] + temp;
           
           // std::cout << expected << "," << result[i][j] << std::endl;

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
  app.run<RangeMappersBench>();
  return 0;
}
