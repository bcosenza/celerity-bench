#include <CL/sycl.hpp>
#include <iostream>

#include "common.h"
#include "bitmap.h"


namespace s = cl::sycl;
class MedianFilterBenchKernel; // kernel forward declaration

void swap(cl::sycl::float4 A[], int i, int j) {
  /*if(A[i] > A[j]) {
    float temp = A[i];
    A[i] = A[j];
    A[j] = temp;
    }*/
  A[i] = fmin(A[i], A[j]);
  A[j] = fmax(A[i], A[j]);
}

/*
  A median filter with a windows of 3 pixels (3x3).
  Input and output are two-dimensional buffers of floats.     
 */
class MedianFilterBench
{
protected:
    std::vector<cl::sycl::float4> input;
    std::vector<cl::sycl::float4> output;

    size_t w, h; // size of the input picture
    size_t size; // user-defined size (input and output will be size x size)
    BenchmarkArgs args;

    PrefetchedBuffer<cl::sycl::float4, 2> input_buf;    
    PrefetchedBuffer<cl::sycl::float4, 2> output_buf;

public:
  MedianFilterBench(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    input.resize(size * size); 
    load_bitmap_mirrored(args.cli.template get<std::string>("--image-file"), size, input);
    output.resize(size * size);

    input_buf.initialize(input.data(), s::range<2>(size, size));    
    output_buf.initialize(output.data(), s::range<2>(size, size));
  }

  void run() {
    celerity::distr_queue& queue = QueueManager::getInstance();

    celerity::buffer<cl::sycl::float4,2>& a = input_buf.get();
    celerity::buffer<cl::sycl::float4,2>& c = output_buf.get();

    queue.submit([=](celerity::handler& cgh) {
      auto in  = a.get_access<s::access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));
      auto out = c.get_access<s::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
      cl::sycl::range<2> ndrange {size, size};

      cgh.parallel_for<class MedianFilterBenchKernel>(ndrange, [in, out, size_ = size](cl::sycl::id<2> gid)
        {
          int x = gid[0];
          int y = gid[1];

          // Optimization note: this array can be prefetched in local memory, TODO
	        cl::sycl::float4 window[9];
          int k = 0;
          for(int i = -1; i<2; i++)
            for(int j = -1; j<2; j++) {
              uint xs = s::min(s::max(x+j, 0), static_cast<int>(size_-1)); // borders are handled here with extended values
              uint ys = s::min(s::max(y+i, 0), static_cast<int>(size_-1));
              window[k] =in[ {xs,ys} ];
              k++;
            }
          
          // (channel-wise) median selection using bitonic sorting
          // the following network is used (Bose-Nelson algorithm):
          // [[0,1],[2,3],[4,5],[7,8]]
          // [[0,2],[1,3],[6,8]]
          // [[1,2],[6,7],[5,8]]
          // [[4,7],[3,8]]
          // [[4,6],[5,7]]
          // [[5,6],[2,7]]
          // [[0,5],[1,6],[3,7]]
          // [[0,4],[1,5],[3,6]]
          // [[1,4],[2,5]]
          // [[2,4],[3,5]]
          // [[3,4]]
          // se also http://pages.ripco.net/~jgamble/nw.html
	        swap(window, 0, 1);
          swap(window, 2, 3);
          swap(window, 0, 2);
          swap(window, 1, 3);
          swap(window, 1, 2);
          swap(window, 4, 5);
          swap(window, 7, 8);
          swap(window, 6, 8);
          swap(window, 6, 7);
          swap(window, 4, 7);
          swap(window, 4, 6);
          swap(window, 5, 8);
          swap(window, 5, 7);
          swap(window, 5, 6);
          swap(window, 0, 5);
          swap(window, 0, 4);
          swap(window, 1, 6);
          swap(window, 1, 5);
          swap(window, 1, 4);
          swap(window, 2, 7);
          swap(window, 3, 8);
          swap(window, 3, 7);
          swap(window, 2, 5);
          swap(window, 2, 4);
          swap(window, 3, 6);
          swap(window, 3, 5);
          swap(window, 3, 4);

	        out[gid] = window[4];
       }
       );
     });
     QueueManager::sync();
     //queue.wait_and_throw();
   }


  bool verify(VerificationSetting &ver) {  
    bool verification_passed = true;

    QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(args.problem_size, args.problem_size));

      for (size_t i = 0; i < args.problem_size; i++) {
        for (size_t j = 0; j < args.problem_size; j++) {
          output[i*args.problem_size + j] = result[i][j];
        }
      }

      cgh.run([=, &verification_passed]() {

        save_bitmap("median.bmp", size, output);

        for(size_t i=ver.begin[0]; i<ver.begin[0]+ver.range[0]; i++){
          int x = i % size;
          int y = i / size;
          cl::sycl::float4 window[9];
          int k = 0;
          for(int i = -1; i<2; i++)
            for(int j = -1; j<2; j++) {
              uint xs = fmin(fmax(x+j, 0), size-1); // borders are handled here with extended values
              uint ys = fmin(fmax(y+i, 0), size-1);
              window[k] = input[xs + ys*size ];
              k++;
            }
          swap(window, 0, 1);
          swap(window, 2, 3);
          swap(window, 0, 2);
          swap(window, 1, 3);
          swap(window, 1, 2);
          swap(window, 4, 5);
          swap(window, 7, 8);
          swap(window, 6, 8);
          swap(window, 6, 7);
          swap(window, 4, 7);
          swap(window, 4, 6);
          swap(window, 5, 8);
          swap(window, 5, 7);
          swap(window, 5, 6);
          swap(window, 0, 5);
          swap(window, 0, 4);
          swap(window, 1, 6);
          swap(window, 1, 5);
          swap(window, 1, 4);
          swap(window, 2, 7);
          swap(window, 3, 8);
          swap(window, 3, 7);
          swap(window, 2, 5);
          swap(window, 2, 4);
          swap(window, 3, 6);
          swap(window, 3, 5);
          swap(window, 3, 4);
          cl::sycl::float4 expected = window[4];
          cl::sycl::float4 dif = fdim(result.get_pointer()[i], expected);
          float length = cl::sycl::length(dif);
          if(length > 0.01f)
          {
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
    return "MedianFilter";
  }

}; // MedianFilterBench class


int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<MedianFilterBench>();  
  return 0;
}


