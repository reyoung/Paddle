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
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
namespace details {

constexpr char kLocalExecScopeName[] = "@LCOAL_SCOPE@";

class OpHandleBase {
 public:
  OpHandleBase() {}

  virtual ~OpHandleBase();

  std::string DebugString() const;

  virtual std::string Name() const = 0;

  void Run(bool use_event);

  virtual void RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx);

  void AddInput(VarHandleBase *in);

  void AddOutput(VarHandleBase *out);

  // This method adds the wait events of all the input on all the device
  // context.
  // NODE: This Wait is asynchronous operation.
  virtual void WaitInputVarGenerated();

  // This method adds the wait events of all the input on the specified device
  // context.
  // NODE: This Wait is asynchronous operation.
  virtual void WaitInputVarGenerated(const platform::Place &place);

  virtual bool NeedWait(VarHandleBase *in_var);

  // If the Op involves data transfer of multiple devices that
  // will likely block other computations.
  virtual bool IsMultiDeviceTransfer() { return false; }

  const platform::DeviceContext *DeviceContext(platform::Place place) {
    return dev_ctxes_[place];
  }

  void SetDeviceContext(platform::Place place, platform::DeviceContext *ctx_) {
    dev_ctxes_[place] = ctx_;
  }

  const std::vector<VarHandleBase *> &Inputs() const { return inputs_; }

  size_t NoDupInputSize() const {
    std::unordered_set<VarHandleBase *> res;
    for (auto *var : inputs_) {
      res.emplace(var);
    }
    return res.size();
  }

  size_t NotReadyInputSize() const {
    std::unordered_set<VarHandleBase *> res;
    for (auto *var : inputs_) {
      if (var->generated_op_ == nullptr) {
        continue;
      }
      res.emplace(var);
    }
    return res.size();
  }

  const std::vector<VarHandleBase *> &Outputs() const { return outputs_; }

 protected:
  void RunAndRecordEvent(const std::function<void()> &callback);

  void RunAndRecordEvent(platform::Place p,
                         const std::function<void()> &callback);

  virtual void RunImpl() = 0;

  std::vector<VarHandleBase *> inputs_;
  std::vector<VarHandleBase *> outputs_;
  std::unordered_map<platform::Place, platform::DeviceContext *,
                     platform::PlaceHash>
      dev_ctxes_;

#ifdef PADDLE_WITH_CUDA
  std::unordered_map<int, cudaEvent_t> events_;
#endif

  DISABLE_COPY_AND_ASSIGN(OpHandleBase);
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
