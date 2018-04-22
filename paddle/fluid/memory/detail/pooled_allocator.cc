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

#include "paddle/fluid/memory/detail/pooled_allocator.h"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>
#include <vector>
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace detail {

class PooledAllocatorPrivate {
 private:
  SystemAllocator* allocator_;
  std::mutex mutex_for_pool_;
  std::map<size_t, std::vector<void*>> pooled_memory;
  size_t align_;
  std::atomic<size_t> memused_;

  std::mutex mutex_for_memsize_;
  std::unordered_map<void*, size_t> memsize_;

 public:
  explicit PooledAllocatorPrivate(SystemAllocator* allocator, size_t align)
      : allocator_(allocator), align_(align), memused_(0) {}

  ~PooledAllocatorPrivate() {
    std::lock_guard<std::mutex> guard(mutex_for_pool_);
    FreePool();
  }

  void* Alloc(size_t unaligned_size) {
    size_t aligned_size = Align(unaligned_size, align_);
    void* res = AllocImpl(aligned_size);
    if (res != nullptr) {
      {
        std::lock_guard<std::mutex> guard(mutex_for_memsize_);
        memsize_.emplace(res, aligned_size);
      }
      memused_ += aligned_size;
    }
    return res;
  }

  void Free(void* ptr) {
    size_t size = 0;
    {
      std::lock_guard<std::mutex> guard(mutex_for_memsize_);
      auto it = memsize_.find(ptr);
      PADDLE_ENFORCE(memsize_.end() != it);
      size = it->second;
      memsize_.erase(it);
    }
    memused_ -= size;
    {
      std::lock_guard<std::mutex> guard(mutex_for_pool_);
      pooled_memory[size].push_back(ptr);
    }
  }

  size_t Used() const { return memused_; }

 private:
  static inline size_t Align(size_t size, size_t alignment) {
    size_t remaining = size % alignment;
    return remaining == 0 ? size : size + (alignment - remaining);
  }

  void* AllocImpl(size_t aligned_size) {
    std::unique_lock<std::mutex> lock(mutex_for_pool_);
    auto it = pooled_memory.lower_bound(aligned_size);
    if (it == pooled_memory.end()) {
      lock.unlock();
      size_t useless = 0;
      auto* res = allocator_->Alloc(&useless, aligned_size);

      if (res != nullptr) {
        return res;
      }
      lock.lock();
      if (pooled_memory.empty()) {
        return nullptr;
      }
      FreePool();
      lock.unlock();
      res = allocator_->Alloc(&useless, aligned_size);
      return res;
    } else {
      auto* retv = it->second.back();
      it->second.pop_back();
      if (it->second.empty()) {
        pooled_memory.erase(it);
      }
      return retv;
    }
  }

  void FreePool() {
    size_t useless = 0;
    for (auto& pair : pooled_memory) {
      for (auto* ptr : pair.second) {
        allocator_->Free(ptr, pair.first, useless);
      }
    }
    pooled_memory.clear();
  }
};

PooledAllocator::PooledAllocator(SystemAllocator* allocator, size_t align)
    : member_(new PooledAllocatorPrivate(allocator, align)) {}

PooledAllocator::~PooledAllocator() { delete member_; }

void* PooledAllocator::Alloc(size_t unaligned_size) {
  return member_->Alloc(unaligned_size);
}

void PooledAllocator::Free(void* ptr) { member_->Free(ptr); }

size_t PooledAllocator::Used() { return member_->Used(); }
}  // namespace detail
}  // namespace memory
}  // namespace paddle
