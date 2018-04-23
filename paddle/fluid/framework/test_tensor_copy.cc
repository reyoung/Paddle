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

#include <thread>  // NOLINT
#include "gtest/gtest.h"

#include "paddle/fluid/framework/lod_tensor.h"

TEST(test_mem_cpy, all) {
  auto fn = [] {
    auto cpu = paddle::platform::CPUPlace();
    paddle::framework::LoDTensor tensor;
    int* cpu_mem = tensor.mutable_data<int>(
        paddle::framework::make_ddim({400, 3, 28, 28}), cpu);
    auto place = paddle::platform::CUDAPlace(0);
    paddle::platform::CUDADeviceContext ctx(place);
    int64_t len = tensor.numel();
    for (int i = 0; i < 10000; ++i) {
      std::fill(cpu_mem, cpu_mem + len, i);
      paddle::framework::LoDTensor gpu_tensor;
      paddle::framework::TensorCopy(tensor, place, ctx, &gpu_tensor);
      paddle::framework::LoDTensor cpu_tensor;
      paddle::framework::TensorCopy(gpu_tensor, cpu, ctx, &cpu_tensor);
      ctx.Wait();
      ASSERT_EQ(cpu_tensor.data<int>()[len - 1], i);
    }
  };

  std::vector<std::thread> threads;

  for (size_t i = 0; i < 1000; ++i) {
    threads.emplace_back(fn);
  }
  for (auto& th : threads) {
    th.join();
  }
}
