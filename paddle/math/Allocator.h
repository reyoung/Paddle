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

#pragma once

#include <stdlib.h>
#include <mutex>
#include "hl_gpu.h"
#include "paddle/utils/Logging.h"

namespace paddle {

/**
 * @brief Allocator base class.
 *
 * This is the base class of all Allocator class.
 */
class Allocator {
public:
  virtual ~Allocator() {}
  virtual void* alloc(size_t size) = 0;
  virtual void free(void* ptr) = 0;
  //  virtual std::string getName() = 0;
};

/**
 * @brief CPU allocator implementation.
 */
class CpuAllocator : public Allocator {
public:
  /**
   * @brief Aligned allocation on CPU.
   * @param size Size to be allocated.
   * @return Pointer to the allocated memory
   */
  void* alloc(size_t size) override {
    void* ptr;
    CHECK_EQ(posix_memalign(&ptr, 32ul, size), 0);
    CHECK(ptr) << "Fail to allocate CPU memory: size=" << size;
    return ptr;
  }

  /**
   * @brief Free the memory space.
   * @param ptr  Pointer to be free.
   */
  void free(void* ptr) override { ::free(ptr); }
};

/**
 * @brief GPU allocator implementation.
 */
class GpuAllocator : public Allocator {
public:
  /**
   * @brief Allocate GPU memory.
   * @param size Size to be allocated.
   * @return Pointer to the allocated memory
   */
  void* alloc(size_t size) override {
    void* ptr = hl_malloc_device(size);
    CHECK(ptr) << "Fail to allocate GPU memory " << size << " bytes";
    return ptr;
  }

  /**
   * @brief Free the GPU memory.
   * @param ptr  Pointer to be free.
   */
  void free(void* ptr) override { hl_free_mem_device(ptr); }
};

/**
 * @brief CPU pinned memory allocator implementation.
 */
class CudaHostAllocator : public Allocator {
public:
  /**
   * @brief Allocate pinned memory.
   * @param size Size to be allocated.
   * @return Pointer to the allocated memory
   */
  void* alloc(size_t size) override {
    void* ptr = hl_malloc_host(size);
    CHECK(ptr) << "Fail to allocate pinned memory " << size << " bytes";
    return ptr;
  }

  /**
   * @brief Free the pinned memory.
   * @param ptr  Pointer to be free.
   */
  void free(void* ptr) override { hl_free_mem_host(ptr); }
};

}  // namespace paddle
