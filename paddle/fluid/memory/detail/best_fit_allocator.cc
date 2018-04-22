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

#include "paddle/fluid/memory/detail/best_fit_allocator.h"
#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace detail {

struct Block;
using list_node_it = std::list<std::unique_ptr<Block>>::iterator;
using tree_node_it = std::multimap<size_t, list_node_it>::iterator;

struct Block {
  void* ptr_{nullptr};
  size_t size_;
  tree_node_it tree_node_;

  explicit Block(size_t size) : size_(size) {}
};

class BestFitAllocatorPrivate {
 private:
  size_t page_size_;

  std::function<void()> dealloc_main_block_;
  std::mutex mutex_;

  // A double linked list of block
  std::list<std::unique_ptr<Block>> block_list_;

  // A rb-tree which store freed blocks.
  // Store the iterator of block_list_.
  std::multimap<size_t, list_node_it> freed_blocks_;

  // A hash table from pointer to iterator of block_list_
  std::unordered_map<void*, list_node_it> used_blocks_;

 public:
  BestFitAllocatorPrivate(SystemAllocator* system_allocator, size_t page_size,
                          size_t pool_size)
      : page_size_(page_size) {
    Block* main = new Block(pool_size);
    size_t useless;
    auto* ptr = system_allocator->Alloc(&useless, pool_size);
    dealloc_main_block_ = [ptr, system_allocator, useless, pool_size] {
      system_allocator->Free(ptr, pool_size, useless);
    };
    main->ptr_ = ptr;
    block_list_.emplace_back(main);
    main->tree_node_ = freed_blocks_.emplace(pool_size, block_list_.begin());
  }

  ~BestFitAllocatorPrivate() { dealloc_main_block_(); }

  // Time complexity O(logN)
  void* Alloc(size_t unaligned_size) {
    size_t aligned_size = Align(unaligned_size, page_size_);
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = freed_blocks_.lower_bound(aligned_size);
    if (it == freed_blocks_.end()) {
      VLOG(10) << "Cannot alloc " << aligned_size;
      return nullptr;
    }

    auto* free_block = it->second->get();

    if (free_block->size_ ==
        aligned_size) {  // special case, free_block is exactly good.
      used_blocks_.emplace(free_block->ptr_, it->second);
      free_block->tree_node_ = freed_blocks_.end();
      freed_blocks_.erase(it);
      return free_block->ptr_;
    }

    auto* new_block = new Block(aligned_size);
    new_block->ptr_ = free_block->ptr_;
    new_block->tree_node_ = freed_blocks_.end();
    reinterpret_cast<int8_t*&>(free_block->ptr_) += aligned_size;
    free_block->size_ -= aligned_size;

    // Insert new_block before it->second.
    used_blocks_.emplace(new_block->ptr_,
                         block_list_.emplace(it->second, new_block));
    return new_block->ptr_;
  }

  // Time complexity O(1)
  void Free(void* ptr) {
    std::lock_guard<std::mutex> guard(mutex_);
    list_node_it to_free_node;
    {
      auto it = used_blocks_.find(ptr);
      PADDLE_ENFORCE(it != used_blocks_.end());
      to_free_node = it->second;
      used_blocks_.erase(it);
    }

    auto* to_free_block = to_free_node->get();

    auto left_node = GetLeftNode(to_free_node);
    auto right_node = GetRightNode(to_free_node);

    bool is_left_free = IsNodeFree(left_node);
    bool is_right_free = IsNodeFree(right_node);
    if (is_left_free && is_right_free) {  // merge left and right
      // Erase right node in freed_block
      auto* left_block = left_node->get();
      auto* right_block = right_node->get();
      left_block->size_ += to_free_block->size_ + right_block->size_;

      // erase to_free_node and right_node in list
      block_list_.erase(right_node);
      block_list_.erase(to_free_node);

      // adjust tree.
      freed_blocks_.erase(left_block->tree_node_);
      freed_blocks_.erase(right_block->tree_node_);
      left_block->tree_node_ =
          freed_blocks_.emplace(left_block->size_, left_node);

    } else if (!is_left_free && is_right_free) {
      // extend right node
      auto* right_block = right_node->get();
      right_block->ptr_ = to_free_node->get()->ptr_;
      right_block->size_ += to_free_node->get()->size_;

      // erase current node
      block_list_.erase(to_free_node);

      // adjust tree.
      freed_blocks_.erase(right_block->tree_node_);
      right_block->tree_node_ =
          freed_blocks_.emplace(right_block->size_, right_node);

    } else if (is_left_free && !is_right_free) {
      // extend left node
      auto* left_block = left_node->get();
      left_block->size_ += to_free_node->get()->size_;

      // erase current node
      block_list_.erase(to_free_node);

      // adjust tree.
      freed_blocks_.erase(left_block->tree_node_);
      left_block->tree_node_ =
          freed_blocks_.emplace(left_block->size_, left_node);

    } else {  // not is_left_free and not is_right free
      // adjust tree
      to_free_block->tree_node_ =
          freed_blocks_.emplace(to_free_block->size_, to_free_node);
    }
  }

  list_node_it GetRightNode(const list_node_it& node_it) {
    auto right_node_it = node_it;
    ++right_node_it;
    return right_node_it;
  }

  list_node_it GetLeftNode(const list_node_it& node_it) {
    auto left_node_it = node_it;
    if (left_node_it == block_list_.begin()) {
      left_node_it = block_list_.end();
    } else {
      --left_node_it;
    }
    return left_node_it;
  }

  bool IsNodeFree(const list_node_it& it) const {
    bool is_node_free =
        it != block_list_.end() && it->get()->tree_node_ != freed_blocks_.end();
    return is_node_free;
  }

 private:
  static inline size_t Align(size_t size, size_t alignment) {
    size_t remaining = size % alignment;
    return remaining == 0 ? size : size + (alignment - remaining);
  }
};

BestFitAllocator::BestFitAllocator(SystemAllocator* system_allocator,
                                   size_t page_size, size_t pool_size)
    : member_(new BestFitAllocatorPrivate(system_allocator, page_size,
                                          pool_size)) {}

BestFitAllocator::~BestFitAllocator() { delete member_; }

void* BestFitAllocator::Alloc(size_t unaligned_size) {
  return member_->Alloc(unaligned_size);
}

void BestFitAllocator::Free(void* ptr) { member_->Free(ptr); }

size_t BestFitAllocator::Used() { return 0; }
}  // namespace detail
}  // namespace memory
}  // namespace paddle
