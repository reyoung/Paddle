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

#include <functional>
#include <memory>
#include <thread>  // NOLINT
#include <vector>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

class AutoIncrementAllocator : public ManagedAllocator {
 public:
  using AllocatorCreator = std::function<std::shared_ptr<ManagedAllocator>()>;

  template <typename Creator>
  explicit AutoIncrementAllocator(Creator&& creator)
      : creator_(std::move(creator)), prev_success_allocator_{0} {}
  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) override;
  std::shared_ptr<Allocation> AllocateShared(size_t size, Attr attr) override;
  bool IsAllocThreadSafe() const override;

 private:
  // NOTE: here use template Callback, it can be inlined when -O3
  template <typename Callback>
  inline typename std::result_of<Callback(ManagedAllocator&)>::type
  InvokeOrCreateUnderlyingAllocator(Callback callback) {
    size_t retry_count = underlying_allocators_.size();
    auto cur = prev_success_allocator_;
    while (retry_count-- > 0) {  // until there retry count is zero
      try {
        auto res = callback(*underlying_allocators_[cur]);
        {
          std::lock_guard<std::mutex> guard(mtx_);
          prev_success_allocator_ = cur;
        }
        return std::move(res);
      } catch (BadAlloc&) {
        ++cur;
        if (cur >= underlying_allocators_.size()) {
          cur = 0;
        }
      } catch (...) {
        // if there is another type of allocation, just rethrow it.
        throw;
      }
    }
    // No suitable allocator
    {
      std::lock_guard<std::mutex> guard(mtx_);
      underlying_allocators_.emplace_back(creator_());
      prev_success_allocator_ = underlying_allocators_.size() - 1;
      return callback(*underlying_allocators_[prev_success_allocator_]);
    }
  }

  AllocatorCreator creator_;
  std::vector<AllocatorCreator::result_type> underlying_allocators_;
  size_t prev_success_allocator_{0};
  std::mutex mtx_;  // NOLINT
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle