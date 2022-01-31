#include <CL/sycl.hpp>
#include <iostream>

#include "common.h"
#include "bitmap.h"


namespace s = cl::sycl;
class SobelBenchKernel; // kernel forward declaration

/*
  A Sobel filter with a convolution matrix 3x3.
  Input and output are two-dimensional buffers of floats.     
 */
class SobelBench
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
  SobelBench(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    input.resize(size * size);
    load_bitmap_mirrored(args.cli.template get<std::string>("--image-file"), size, input);
    output.resize(size * size);

    auto range = celerity::range<2>(size, size);
    input_buf.initialize(input.data(), range);    
    output_buf.initialize(output.data(), range);
  }

  void run() {
    celerity::distr_queue& queue = QueueManager::getInstance();

    celerity::buffer<cl::sycl::float4,2>& a = input_buf.get();
    celerity::buffer<cl::sycl::float4,2>& c = output_buf.get();

    queue.submit([=](celerity::handler& cgh) {
      celerity::accessor in{a, cgh, celerity::access::neighborhood<2>(1,1), celerity::read_only};
      celerity::accessor out{c, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
      cl::sycl::range<2> ndrange{size, size};

      // Sobel kernel 3x3
      const float kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

      cgh.parallel_for<class SobelBenchKernel>(ndrange, [in, out, kernel, size_ = size](cl::sycl::id<2> gid) {
        int x = gid[0];
        int y = gid[1];
        cl::sycl::float4 Gx = cl::sycl::float4(0, 0, 0, 0);
        cl::sycl::float4 Gy = cl::sycl::float4(0, 0, 0, 0);
        const int radius = 3;

        // constant-size loops in [0,1,2]
        for(int x_shift = 0; x_shift < 3; x_shift++) {
          for(int y_shift = 0; y_shift < 3; y_shift++) {
            // sample position
            uint xs = x + x_shift - 1; // [x-1,x,x+1]
            uint ys = y + y_shift - 1; // [y-1,y,y+1]
            // for the same pixel, convolution is always 0
            if(x == xs && y == ys)
              continue;
            // boundary check
            if(xs < 0 || xs >= size_ || ys < 0 || ys >= size_)
              continue;

            // sample color
            cl::sycl::float4 sample = in[{xs, ys}];

            // convolution calculation
            int offset_x = x_shift + y_shift * radius;
            int offset_y = y_shift + x_shift * radius;

            float conv_x = kernel[offset_x];
            cl::sycl::float4 conv4_x = cl::sycl::float4(conv_x);
            Gx += conv4_x * sample;

            float conv_y = kernel[offset_y];
            cl::sycl::float4 conv4_y = cl::sycl::float4(conv_y);
            Gy += conv4_y * sample;
          }
        }
        // taking root of sums of squares of Gx and Gy
        cl::sycl::float4 color = hypot(Gx, Gy);
        cl::sycl::float4 minval = cl::sycl::float4(0.0, 0.0, 0.0, 0.0);
        cl::sycl::float4 maxval = cl::sycl::float4(1.0, 1.0, 1.0, 1.0);
        out[gid] = clamp(color, minval, maxval);
      });
    });
   }


  bool verify(VerificationSetting& ver) {
    bool pass = true;
    /*QueueManager::getInstance().with_master_access([&](celerity::handler& cgh) {
      auto result = output_buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(args.problem_size, args.problem_size));
      
      for (size_t i = 0; i < args.problem_size; i++) {
        for (size_t j = 0; j < args.problem_size; j++) {
          output[i*args.problem_size + j] = result[i][j];
        }
      }

      cgh.run([=, &pass]() {

        save_bitmap("sobel3.bmp", size, output);

        const float kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
        int radius = 3;
        for(size_t i = ver.begin[0]; i < ver.begin[0] + ver.range[0]; i++) {
          int x = i % size;
          int y = i / size;
          cl::sycl::float4 Gx, Gy;
          for(uint x_shift = 0; x_shift < 3; x_shift++)
            for(uint y_shift = 0; y_shift < 3; y_shift++) {
              uint xs = x + x_shift - 1;
              uint ys = y + y_shift - 1;
              if(x == xs && y == ys)
                continue;
              if(xs < 0 || xs >= size || ys < 0 || ys >= size)
                continue;
              cl::sycl::float4 sample = input[xs + ys * size];
              int offset_x = x_shift + y_shift * radius;
              int offset_y = y_shift + x_shift * radius;
              float conv_x = kernel[offset_x];
              cl::sycl::float4 conv4_x = cl::sycl::float4(conv_x);
              Gx += conv4_x * sample;
              float conv_y = kernel[offset_y];
              cl::sycl::float4 conv4_y = cl::sycl::float4(conv_y);
              Gy += conv4_y * sample;
            }

          cl::sycl::float4 color = hypot(Gx, Gy);
          cl::sycl::float4 minval = cl::sycl::float4(0.0, 0.0, 0.0, 0.0);
          cl::sycl::float4 maxval = cl::sycl::float4(1.0, 1.0, 1.0, 1.0);
          cl::sycl::float4 expected = clamp(color, minval, maxval);
          cl::sycl::float4 dif = fdim(output[i], expected);
          float length = cl::sycl::length(dif);
          if(length > 0.01f) {
            pass = false;
            break;
          }
        } 
      });
    });*/
    QueueManager::sync();
    return pass;
  }


  static std::string getBenchmarkName() { return "Sobel3"; }

}; // SobelBench class


int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<SobelBench>();  
  return 0;
}


