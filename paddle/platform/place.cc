/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/platform/place.h"

namespace paddle {
namespace platform {

namespace detail {

class PlacePrinter : public boost::static_visitor<> {
 public:
  explicit PlacePrinter(std::ostream &os) : os_(os) {}
  void operator()(const CPUPlace &) { os_ << "CPUPlace"; }
  void operator()(const CUDAPlace &p) {
    os_ << "CUDAPlace(" << p.device << ")";
  }

 private:
  std::ostream &os_;
};

}  // namespace detail

static Place the_default_place;

void set_place(const Place &place) { the_default_place = place; }
const Place &get_place() { return the_default_place; }

const CUDAPlace default_gpu() { return CUDAPlace(0); }
const CPUPlace default_cpu() { return CPUPlace(); }

bool is_gpu_place(const Place &p) {
  return boost::apply_visitor(IsCUDAPlace(), p);
}

bool is_cpu_place(const Place &p) { return !is_gpu_place(p); }

bool places_are_same_class(const Place &p1, const Place &p2) {
  return p1.which() == p2.which();
}

bool is_same_place(const Place &p1, const Place &p2) {
  if (places_are_same_class(p1, p2)) {
    if (is_cpu_place(p1)) {
      return true;
    } else {
      return boost::get<CUDAPlace>(p1) == boost::get<CUDAPlace>(p2);
    }
  } else {
    return false;
  }
}

std::ostream &operator<<(std::ostream &os, const Place &p) {
  detail::PlacePrinter printer(os);
  boost::apply_visitor(printer, p);
  return os;
}

CUDAPlaceGuard::CUDAPlaceGuard(const CUDAPlace place) {
  (void)(this->dev_id_);  // ignore gcc unused warning in CPU
#ifndef PADDLE_WITH_CUDA
  PADDLE_THROW(
      "Should not invoke CUDAPlaceGuard when paddle is not compiled with CUDA");
#else
  this->dev_id_ = cudaGetDevice();
  cudaSetDevice(place.device);
#endif
}

CUDAPlaceGuard::~CUDAPlaceGuard() {
#ifdef PADDLE_WITH_CUDA
  cudaSetDevice(this->dev_id_);
#endif
}

}  // namespace platform
}  // namespace paddle
