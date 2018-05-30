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

int main() {
  InitDevices(true);

  Scope scope;
  auto& new_scope = scope.NewScope();
  std::vector<std::string> vars;

  for (int i = 0; i < GetCUDADeviceCount(); ++i) {
    vars.emplace_back();
    new_scope.Var(&vars.back())->GetMutable<LoDTensor>();
  }

  *scope.Var(details::kLocalExecScopeName)->GetMutable<Scope*>() = &new_scope;

  BlockingQueue<std::string> job;
  job.Extend(vars);

  size_t counter = 0;
  size_t num_iteration = 100000;
  std::mutex counter_mtx;
  std::condition_variable counter_cv;

  std::vector<std::thread> thread;
  for (size_t i = 0; i < vars.size(); ++i) {
    thread.emplace_back(
        [i, &counter, &num_iteration, &counter_mtx, &counter_cv, &scope, &job] {
          OpDesc desc;
          desc.SetType("uniform_random");
          desc.SetAttr("shape", std::vector<int>{64, 224, 224});

          while (true) {
            auto name = job.Pop();
            if (name.empty()) {
              break;
            }
            desc.SetOutput("Out", {name});

            details::ComputationOpHandle op_handle(desc, &scope, CUDAPlace(i));
            op_handle.SetDeviceContext(
                CUDAPlace(i), DeviceContextPool::Instance().Get(CUDAPlace(i)));
            op_handle.Run(true);
            bool at_end;
            {
              std::lock_guard<std::mutex> guard(counter_mtx);
              ++counter;
              at_end = counter == num_iteration;
            }
            if (!at_end) {
              job.Push(name);
            }
            counter_cv.notify_one();
          }
        });
  }

  std::unique_lock<std::mutex> lock(counter_mtx);
  counter_cv.wait(lock, [&] { return counter == num_iteration; });

  for (size_t i = 0; i < vars.size(); ++i) {
    job.Push("");
  }

  for (auto& th : thread) {
    th.join();
  }

  return 0;
}
