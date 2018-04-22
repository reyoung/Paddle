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

#include "paddle/fluid/memory/detail/allocator_base.h"

namespace paddle {
namespace memory {
namespace detail {

class SystemAllocator;

class BestFitAllocatorPrivate;
// A best fit allocator, the time complexity of Alloc/Free are O(logN)/O(1).
// https://www.boost.org/doc/libs/1_62_0/doc/html/interprocess/memory_algorithms.html#interprocess.memory_algorithms.rbtree_best_fit
class BestFitAllocator : public AllocatorBase {
 public:
  BestFitAllocator(SystemAllocator* system_allocator, size_t page_size,
                   size_t pool_size);
  ~BestFitAllocator();

  void* Alloc(size_t unaligned_size) override;

  void Free(void* ptr) override;

  size_t Used() override;

 private:
  BestFitAllocatorPrivate* member_;
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
