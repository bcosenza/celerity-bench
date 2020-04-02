#pragma once
#include <celerity/celerity.h>
#include <memory>

template<class AccType>
class InitializationDummyKernel
{
public:
  InitializationDummyKernel(AccType acc)
  : acc{acc} {}

  void operator()() {}
private:
  AccType acc;
};

class InitializationDummyKernel2;

template <class T, int Dimensions=1>
class PrefetchedBuffer {
public:
  void initialize(cl::sycl::range<Dimensions> r) {
    buff = std::make_shared<celerity::buffer<T, Dimensions>>(r);
  }

  void initialize(T* data, cl::sycl::range<Dimensions> r) {
    buff = std::make_shared<celerity::buffer<T, Dimensions>>(data, r);
  }

  void initialize(const T* data, cl::sycl::range<Dimensions> r) {
    buff = std::make_shared<celerity::buffer<T, Dimensions>>(data, r);
  }

  template <cl::sycl::access::mode mode>
  auto get_access(celerity::handler& commandGroupHandler, cl::sycl::range<Dimensions> accessRange) {
    return buff->template get_access<mode>(commandGroupHandler, accessRange);
  }

  /* TODO - untested, uncomment and adapt if needed
  template <cl::sycl::access::mode mode, cl::sycl::access::target target = cl::sycl::access::target::global_buffer>
  auto get_access(cl::sycl::handler& commandGroupHandler) {
    return buff->template get_access<mode, target>(commandGroupHandler);
  }

  template <cl::sycl::access::mode mode>
  auto get_access() {
    return buff->template get_access<mode>();
  }

  template <cl::sycl::access::mode mode, cl::sycl::access::target target = cl::sycl::access::target::global_buffer>
  auto get_access(cl::sycl::handler& commandGroupHandler, cl::sycl::range<Dimensions> accessRange,
      cl::sycl::id<Dimensions> accessOffset = {}) {
    return buff->template get_access<mode, target>(commandGroupHandler, accessRange, accessOffset);
  }

  template <cl::sycl::access::mode mode>
  auto get_access(cl::sycl::range<Dimensions> accessRange, cl::sycl::id<Dimensions> accessOffset = {}) {
    return buff->template get_access<mode>(accessRange, accessOffset);
  }
  */

  cl::sycl::range<Dimensions> get_range() const
  {
    return buff->get_range();
  }

  celerity::buffer<T, Dimensions>& get() const { return *buff; }

  void reset() { buff = nullptr; }

private:
  // Wrap in a shared_ptr to allow default constructing this class
  std::shared_ptr<celerity::buffer<T, Dimensions>> buff;
};
