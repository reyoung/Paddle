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

#include <thread>  // NOLINT
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/init.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

using namespace paddle::framework;  // NOLINT
using namespace paddle::platform;   // NOLINT

USE_OP(uniform_random);
USE_OP(conv2d);
USE_OP_DEVICE_KERNEL(conv2d, CUDNN);

struct JobItem {
  std::string in;
  std::string filter;
  std::string out;
  int dev_id;
};

int main() {
  InitDevices(true);

  Scope scope;
  auto& new_scope = scope.NewScope();
  std::vector<std::string> input_vars;
  std::vector<std::string> filter_vars;
  std::vector<std::string> out_vars;
  BlockingQueue<JobItem> job;

  for (int i = 0; i < GetCUDADeviceCount(); ++i) {
    input_vars.emplace_back();
    new_scope.Var(&input_vars.back())
        ->GetMutable<LoDTensor>()
        ->Resize({40, 128, 14, 14})
        .mutable_data<float>(CUDAPlace(i));

    filter_vars.emplace_back();
    new_scope.Var(&filter_vars.back())
        ->GetMutable<LoDTensor>()
        ->Resize({128, 128, 3, 3})
        .mutable_data<float>(CUDAPlace(i));

    out_vars.emplace_back();
    new_scope.Var(&out_vars.back())->GetMutable<LoDTensor>();

    job.Push({input_vars.back(), filter_vars.back(), out_vars.back(), i});
  }

  *scope.Var(details::kLocalExecScopeName)->GetMutable<Scope*>() = &new_scope;

  size_t counter = 0;
  size_t num_iteration = 100000;
  std::mutex counter_mtx;
  std::condition_variable counter_cv;

  std::vector<std::thread> thread;
  for (size_t i = 0; i < input_vars.size(); ++i) {
    thread.emplace_back([&] {
      OpDesc desc;
      desc.SetType("conv2d");
      desc.SetAttr("use_cudnn", true);
      while (true) {
        auto item = job.Pop();
        if (item.dev_id == -1) {
          break;
        }
        desc.SetOutput("Output", {item.out});
        desc.SetInput("Input", {item.in});
        desc.SetInput("Filter", {item.filter});

        details::ComputationOpHandle op_handle(desc, &scope,
                                               CUDAPlace(item.dev_id));
        op_handle.SetDeviceContext(
            CUDAPlace(item.dev_id),
            DeviceContextPool::Instance().Get(CUDAPlace(item.dev_id)));
        op_handle.Run(true);
        bool at_end;
        {
          std::lock_guard<std::mutex> guard(counter_mtx);
          ++counter;
          at_end = counter == num_iteration;
        }
        if (!at_end) {
          job.Push(item);
        }
        counter_cv.notify_one();
      }
    });
  }

  std::unique_lock<std::mutex> lock(counter_mtx);
  counter_cv.wait(lock, [&] { return counter == num_iteration; });

  for (size_t i = 0; i < input_vars.size(); ++i) {
    job.Push({"", "", "", -1});
  }

  for (auto& th : thread) {
    th.join();
  }

  return 0;
}
