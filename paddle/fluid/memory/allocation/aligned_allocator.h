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

#pragma once
#include <memory>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

// The aligned allocation and allocator will wrap a managed allocator,
// and returns the aligned pointer.
//
// NOTE(yy): For speed reason, I just use a template parameter to get
// alignment, however, it can be an private member if necessary.
//
// NOTE(yy): kAlignment must be 2^N. a `static_assert` should be added.
template <size_t kAlignment>
class AlignedAllocation : public Allocation {
  static_assert(kAlignment > 0 && (kAlignment & (kAlignment - 1)) == 0,
                "kAlignment must be 2^N");

 public:
  AlignedAllocation(std::unique_ptr<Allocation>&& underlying_allocation,
                    size_t size)
      : Allocation(AlignedPtr(underlying_allocation->ptr()),
                   size + kAlignment - Offset(underlying_allocation->ptr()),
                   underlying_allocation->place()),
        underlying_allocation_(std::move(underlying_allocation)) {}

 private:
  static void* AlignedPtr(void* ptr) {
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) +
                                   Offset(ptr));
  }

  // Offset to aligned pointer.
  // if ptr is already aligned, returns 0.
  static size_t Offset(void* ptr) {
    auto ptr_addr = reinterpret_cast<intptr_t>(ptr);
    intptr_t aligned_addr = (ptr_addr & ~(kAlignment - 1));
    intptr_t diff = aligned_addr - ptr_addr;
    if (diff == 0) {
      return 0;
    } else {
      return kAlignment + diff;
    }
  }

  std::unique_ptr<Allocation> underlying_allocation_;
};

// Thin aligned allocator is trivial and used to generate a small size binary.
//
// NOTE(yy): This is a trick to make a template class. This class extract the
// common code into a `thin` class. So if there are multiple specification of
// the template class, the binary size will not extended too much.
//
// NOTE(yy): This could be an over design. If it harms readability of code, it
// could be removed later.
class ThinAlignedAllocator : public ManagedAllocator {
 public:
  explicit ThinAlignedAllocator(
      std::shared_ptr<ManagedAllocator> underlyning_allocator);

  std::shared_ptr<Allocation> AllocateShared(size_t size, Attr attr) override;

  bool IsAllocThreadSafe() const;

 protected:
  std::shared_ptr<ManagedAllocator> underlying_allocator_;
};

// An aligned allocator will allocate `size+kAlignment` allocation and adjust
// the pointer offset.
template <size_t kAlignment>
class AlignedAllocator : public ThinAlignedAllocator {
 public:
  using ThinAlignedAllocator::ThinAlignedAllocator;
  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) override {
    auto raw_allocation =
        underlying_allocator_->Allocate(size + kAlignment, attr);
    return std::unique_ptr<Allocation>(
        new AlignedAllocation<kAlignment>(std::move(raw_allocation), size));
  }
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle