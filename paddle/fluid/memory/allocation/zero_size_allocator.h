// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <utility>

#pragma once

#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

// The allocator handles the request's size is zero. Allocator will always
// return an allocation even the request size is zero. However, the
// allocation.ptr() is nullptr
class ZeroSizeAllocation : public Allocation {
 public:
  explicit ZeroSizeAllocation(const platform::Place& p)
      : Allocation(nullptr, 0, p) {}
};

class ZeroSizeAllocator : public ManagedAllocator {
 public:
  ZeroSizeAllocator(
      const std::shared_ptr<ManagedAllocator>& underlying_allocator,
      const platform::Place& p)
      : underlying_allocator_(underlying_allocator), place_(p) {}
  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) override;
  std::shared_ptr<Allocation> AllocateShared(size_t size, Attr attr) override;
  bool IsAllocThreadSafe() const override;

 private:
  std::shared_ptr<ManagedAllocator> underlying_allocator_;
  const platform::Place& place_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle