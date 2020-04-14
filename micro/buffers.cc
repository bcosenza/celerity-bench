#include "common.h"

#include <iostream>
namespace s = cl::sycl;

template <typename T> class TwoBuffersOneToOneMapperKernel;
template <typename T> class FourBuffersOneToOneMapperKernel;
template <typename T> class SixBuffersOneToOneMapperKernel;
template <typename T> class TwoBuffersAllMapperKernel;
template <typename T> class FourBuffersAllMapperKernel;
template <typename T> class SixBuffersAllMapperKernel;

template <typename T>
class RangeMappersBench
{
protected:    
  std::vector<T> input1;
  std::vector<T> input2;
  std::vector<T> input3;
  std::vector<T> input4;
  std::vector<T> input5;
  std::vector<T> input6;    
  std::vector<T> output;
  BenchmarkArgs args;

  PrefetchedBuffer<T, 2> input1_buf;
  PrefetchedBuffer<T, 2> input2_buf;
  PrefetchedBuffer<T, 2> input3_buf;
  PrefetchedBuffer<T, 2> input4_buf;
  PrefetchedBuffer<T, 2> input5_buf;
  PrefetchedBuffer<T, 2> input6_buf;
  PrefetchedBuffer<T, 2> output_buf;

public:
  RangeMappersBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {
    // host memory intilization
    input1.resize(args.problem_size*args.problem_size);
    input2.resize(args.problem_size*args.problem_size);
    input3.resize(args.problem_size*args.problem_size);
    input4.resize(args.problem_size*args.problem_size);
    input5.resize(args.problem_size*args.problem_size);
    input6.resize(args.problem_size*args.problem_size);
    output.resize(args.problem_size*args.problem_size);

    for (size_t i = 0; i < args.problem_size*args.problem_size; i++) {
      input1[i] = static_cast<T>(i);
      input2[i] = static_cast<T>(i);
      input3[i] = static_cast<T>(i);
      input4[i] = static_cast<T>(i);
      input5[i] = static_cast<T>(i);
      input6[i] = static_cast<T>(i);            
      output[i] = static_cast<T>(0);
    }

    input1_buf.initialize(input1.data(), s::range<2>(args.problem_size, args.problem_size));
    input2_buf.initialize(input2.data(), s::range<2>(args.problem_size, args.problem_size));
    input3_buf.initialize(input3.data(), s::range<2>(args.problem_size, args.problem_size));
    input4_buf.initialize(input4.data(), s::range<2>(args.problem_size, args.problem_size));
    input5_buf.initialize(input5.data(), s::range<2>(args.problem_size, args.problem_size));
    input6_buf.initialize(input6.data(), s::range<2>(args.problem_size, args.problem_size)); 

    output_buf.initialize(output.data(), s::range<2>(args.problem_size, args.problem_size));
  }

  void two_buffers_map121(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class TwoBuffersOneToOneMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  }

  void four_buffers_map121(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 
  celerity::buffer<T, 2>& buf_c, celerity::buffer<T, 2>& buf_d,celerity::buffer<T,2>& buf_e) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto d = buf_d.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());      
      auto e = buf_e.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class FourBuffersOneToOneMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        e[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}] + 
                                c[{item[0], item[1]}] + d[{item[0], item[1]}];
      });
    });
  } 

  void six_buffers_map121(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 
  celerity::buffer<T, 2>& buf_c, celerity::buffer<T, 2>& buf_d, celerity::buffer<T, 2>& buf_e, celerity::buffer<T, 2>& buf_f, 
  celerity::buffer<T,2>& buf_g) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto d = buf_d.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto e = buf_e.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());
      auto f = buf_f.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<2>());            
      auto g = buf_g.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

      cgh.parallel_for<class SixBuffersOneToOneMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        g[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}] +
                                c[{item[0], item[1]}] + d[{item[0], item[1]}] +
                                e[{item[0], item[1]}] + f[{item[0], item[1]}];
      });
    });
  }   

  void two_buffers_mapAll(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 2>& buf_c) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::all<2>());

      cgh.parallel_for<class TwoBuffersAllMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        c[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}];
      });
    });
  } 

  void four_buffers_mapAll(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 
  celerity::buffer<T, 2>& buf_c, celerity::buffer<T, 2>& buf_d,celerity::buffer<T,2>& buf_e) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto d = buf_d.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());      
      auto e = buf_e.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::all<2>());

      cgh.parallel_for<class FourBuffersAllMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        e[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}] + 
                                c[{item[0], item[1]}] + d[{item[0], item[1]}];
      });
    });
  } 

  void six_buffers_mapAll(celerity::distr_queue& queue, celerity::buffer<T, 2>& buf_a, celerity::buffer<T, 2>& buf_b,celerity::buffer<T, 
  celerity::buffer<T, 2>& buf_c, celerity::buffer<T, 2>& buf_d, celerity::buffer<T, 2>& buf_e, celerity::buffer<T, 2>& buf_f, 
  celerity::buffer<T,2>& buf_g) {
    queue.submit([=](celerity::handler& cgh) {
      auto a = buf_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto b = buf_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto c = buf_c.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto d = buf_d.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto e = buf_e.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());
      auto f = buf_f.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());            
      auto g = buf_g.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::all<2>());

      cgh.parallel_for<class SixBuffersAllMapperKernel<T>>(cl::sycl::range<2>(args.problem_size, args.problem_size), [=](cl::sycl::item<2> item) {
        g[{item[0], item[1]}] = a[{item[0], item[1]}] + b[{item[0], item[1]}] +
                                c[{item[0], item[1]}] + d[{item[0], item[1]}] +
                                e[{item[0], item[1]}] + f[{item[0], item[1]}];
      });
    });
  }            

  void run() {
  
    // Matrix addition using one_to_one ranage mapper
    two_buffers_map121(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), 
    input3_buf.get(), input4_buf.get(), input5_buf.get(), input6_buf.get(), output_buf.get());

    // Matrix addition using one_to_one ranage mapper
    four_buffers_map121(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), 
    input3_buf.get(), input4_buf.get(), output_buf.get());
    
    // Matrix addition using one_to_one ranage mapper
    six_buffers_map121(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), 
    input3_buf.get(), input4_buf.get(), input5_buf.get(), input6_buf.get(), output_buf.get());         

    // Matrix addition using all ranage mapper
    two_buffers_mapAll(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), output_buf.get());

    // Matrix addition using all ranage mapper
    four_buffers_mapAll(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), 
     input3_buf.get(), input4_buf.get(), output_buf.get());   

    // Matrix addition using all ranage mapper
    six_buffers_mapAll(QueueManager::getInstance(), input1_buf.get(), input2_buf.get(), 
    input3_buf.get(), input4_buf.get(), input5_buf.get(), input6_buf.get(), output_buf.get());     

  }

  bool verify(VerificationSetting &ver) {
    bool verification_passed = true;
    QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(args.problem_size, args.problem_size));
      cgh.run([=, &verification_passed]() {
        for(size_t i = 0; i < args.problem_size; i++){
          for (size_t j = 0; j < args.problem_size; j++) {
            auto expected = input1[i*args.problem_size + j] + input2[i*args.problem_size + j];
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
