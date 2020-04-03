#include "common.h"    
#include <iostream>
#include <type_traits>
#include <iomanip>

//#define parallelForWorkGroup

//using namespace cl::sycl;
namespace s = cl::sycl;

template<typename T>
class ScalarProdKernel;

template<typename T>
class ScalarProdReduction;

template<typename T>
class ScalarProdBench
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
  ScalarProdBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    // host memory allocation and initialization
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    output.resize(args.problem_size);

    for (size_t i = 0; i < args.problem_size; i++) {
      input1[i] = static_cast<T>(1);
      input2[i] = static_cast<T>(2);
      output[i] = static_cast<T>(0);
    }

    input1_buf.initialize(input1.data(), s::range<1>(args.problem_size));
    input2_buf.initialize(input2.data(), s::range<1>(args.problem_size));
    output_buf.initialize(output.data(), s::range<1>(args.problem_size));
  }

  void run() {

    celerity::distr_queue& queue = QueueManager::getInstance();

    celerity::buffer<T,1>& a = input1_buf.get();
    celerity::buffer<T,1>& b = input2_buf.get();
    celerity::buffer<T,1>& c = output_buf.get();
    
    queue.submit([=](celerity::handler& cgh) {
      auto in1 = a.template get_access<s::access::mode::read>(cgh, celerity::access::one_to_one<1>());
      auto in2 = b.template get_access<s::access::mode::read>(cgh, celerity::access::one_to_one<1>());
      // Use discard_write here, otherwise the content of the hostbuffer must first be copied to device
      auto intermediate_product = c.template get_access<s::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());

      cl::sycl::range<1> ndrange (args.problem_size);

      cgh.parallel_for<class ScalarProdKernel<T>>(ndrange,
        [=](cl::sycl::id<1> item) 
        {
          size_t gid= item.get_global_linear_id();
          intermediate_product[gid] = in1[gid] * in2[gid];
        });
    });

    // std::cout << "Multiplication of vectors completed" << std::endl;

    auto array_size = args.problem_size;
    auto wgroup_size = args.local_size;
    // Not yet tested with more than 2
    auto elements_per_thread = 2;

    while (array_size!= 1) {
      auto n_wgroups = (array_size + wgroup_size*elements_per_thread - 1)/(wgroup_size*elements_per_thread); // two threads per work item

      queue.submit([=](celerity::handler& cgh) {

          auto global_mem = c.template get_access<s::access::mode::read_write>(cgh, celerity::access::one_to_one<1>());
      
          // local memory for reduction
          auto local_mem = s::accessor <T, 1, s::access::mode::read_write, s::access::target::local> {s::range<1>(wgroup_size), cgh, celerity::access::one_to_one<1>()};
          cl::sycl::range<1> ndrange (n_wgroups*wgroup_size);
    

          cgh.parallel_for<class ScalarProdReduction<T>>(ndrange,
          [=](cl::sycl::id<1> item) 
            {
              size_t gid= item.get_global_linear_id();
              size_t lid = item.get_local_linear_id();

              // initialize local memory to 0
              local_mem[lid] = 0; 

              if ((elements_per_thread * gid) < array_size) {
                  local_mem[lid] = global_mem[elements_per_thread*gid] + global_mem[elements_per_thread*gid + 1];
              }

              item.barrier(s::access::fence_space::local_space);

              for (size_t stride = 1; stride < wgroup_size; stride *= elements_per_thread) {
                auto local_mem_index = elements_per_thread * stride * lid;
                if (local_mem_index < wgroup_size) {
                    local_mem[local_mem_index] = local_mem[local_mem_index] + local_mem[local_mem_index + stride];
                }

                item.barrier(s::access::fence_space::local_space);
              }

              // Only one work-item per work group writes to global memory 
              if (lid == 0) {
                global_mem[item.get_group_linear_id()] = local_mem[0];
              }
            });
        });

      array_size = n_wgroups;
    }
  }

  bool verify(VerificationSetting &ver) { 
    bool pass = true;
    auto expected = static_cast <T>(0);

   // celerity::buffer<T,1>& c = output_buf.get();
    auto output_acc = output_buf.template get_access<s::access::mode::read>();

    for(size_t i = 0; i < args.problem_size; i++) {
        expected += input1[i] * input2[i];
    }

    //std::cout << "Scalar product on CPU =" << expected << std::endl;
    //std::cout << "Scalar product on Device =" << output[0] << std::endl;

    // Todo: update to type-specific test (Template specialization?)
    const auto tolerance = 0.00001f;
    if(std::fabs(expected - output_acc[0]) > tolerance) {
      pass = false;
    }

    return pass;
  }
  
  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "ScalarProduct_";
   // name << (Use_ndrange ? "NDRange_" : "Hierarchical_");
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<ScalarProdBench<int>>();
  app.run<ScalarProdBench<long long>>();
  app.run<ScalarProdBench<float>>();
  app.run<ScalarProdBench<double>>();

  return 0;
}
