//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cstdlib>  // for size_t

namespace paddle {
namespace memory {
namespace detail {

class AllocatorBase {
 public:
  AllocatorBase() = default;
  virtual ~AllocatorBase() = default;
  virtual void* Alloc(size_t unaligned_size) = 0;
  virtual void Free(void* ptr) = 0;
  virtual size_t Used() = 0;

  // Disable copy and assignment
  AllocatorBase(const AllocatorBase&) = delete;
  AllocatorBase& operator=(const AllocatorBase&) = delete;
};
}  // namespace detail
}  // namespace memory
}  // namespace paddle
