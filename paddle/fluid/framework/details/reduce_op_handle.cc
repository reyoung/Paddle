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

#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/gather_op_handle.h"
#include "paddle/fluid/framework/details/reduce_util.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace framework {
namespace details {

ReduceOpHandle::ReduceOpHandle(const std::vector<Scope *> &local_scopes,
                               const std::vector<platform::Place> &places,
                               const platform::NCCLContextMap &ctxs)
    : local_scopes_(local_scopes), places_(places), nccl_ctxs_(ctxs) {}

void ReduceOpHandle::RunImpl() {
  // the input may have dummy var.
  std::vector<VarHandle *> in_var_handles;
  for (auto *in : inputs_) {
    auto *in_handle = dynamic_cast<VarHandle *>(in);
    if (in_handle) {
      in_var_handles.push_back(in_handle);
    }
  }
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), places_.size(),
      "The number of output should equal to the number of places.");

  // the output may have dummy var.
  std::vector<VarHandle *> out_var_handles;
  for (auto *out : outputs_) {
    auto *out_handle = dynamic_cast<VarHandle *>(out);
    if (out_handle) {
      out_var_handles.push_back(out_handle);
    }
  }
  PADDLE_ENFORCE_EQ(out_var_handles.size(), 1,
                    "The number of output should be one.");

  auto in_0_handle = static_cast<VarHandle *>(in_var_handles[0]);
  auto pre_in_var =
      local_scopes_[in_0_handle->scope_idx_]->FindVar(in_0_handle->name_);
  auto pre_place = in_0_handle->place_;

  // TODO(zcd): add wait

  if (pre_in_var->IsType<framework::SelectedRows>()) {
    // gather
    std::unique_ptr<OpHandleBase> op_handle_;
    op_handle_.reset(new GatherOpHandle(local_scopes_, places_));
    op_handle_->dev_ctxes_ = dev_ctxes_;
    op_handle_->inputs_ = inputs_;
    op_handle_->outputs_ = outputs_;
    //    op_handle_

  } else {
    auto pre_in = pre_in_var->Get<framework::LoDTensor>();
    std::vector<LoDTensor> lod_tensors;
    std::vector<platform::Place> &in_places;
    for (auto *in : in_var_handles) {
      auto in_handle = static_cast<VarHandle *>(in);
      auto in_p = in_handle->place_;
      auto in_var =
          local_scopes_.at(in_handle->scope_idx_)->FindVar(in_handle->name_);
      auto &in_sr = in_var->Get<framework::LoDTensor>();

      PADDLE_ENFORCE_EQ(in_p.which(), pre_place.which(),
                        "Places must be all on CPU or all on CUDA.");
      PADDLE_ENFORCE_EQ(in_sr.type(), pre_in.type(),
                        "The type of input is not consistent.");

      in_places.emplace_back(in_p);
      lod_tensors.emplace_back(in_sr.value());
    }

    auto &trg = local_scopes_[out_var_handles[0]->scope_idx_]
                    ->FindVar(out_var_handles[0]->name_)
                    ->GetMutable<framework::LoDTensor>();
    trg->Resize(pre_in.dims());
    trg->mutable_data(out_var_handles[0]->place_, pre_in.type());

    if (paddle::platform::is_cpu_place(pre_place)) {
      ReduceLoDTensor func(lod_tensors, &trg);
      VisitDataType(ToDataType(lod_tensors[0].type()), func);
    } else if (paddle::platform::is_gpu_place(pre_place)) {
// nccl reduce
#ifdef PADDLE_WITH_CUDA
#else
#endif
      int root =
          static_cast<platform::CUDAPlace>(out_var_handles[0]->place_).device;

      std::vector<std::function<void()>> all_reduce_calls;
      for (size_t i = 0; i < local_scopes_.size(); ++i) {
        auto &p = places_[i];
        auto &lod_tensor = lod_tensors[i];

        void *buffer = const_cast<void *>(lod_tensor.data<void>());

        if (dtype == -1) {
          dtype = platform::ToNCCLDataType(lod_tensor.type());
        }

        if (numel == 0) {
          numel = static_cast<size_t>(lod_tensor.numel());
        }

        T *recvbuffer = nullptr;
        if (root == gpu_id) {
          recvbuffer = trg->mutable_data(out_var_handles[0]->place_);
        } else {
          out->Resize(framework::make_ddim({0}));
        }

        int dev_id = boost::get<platform::CUDAPlace>(p).device;
        auto &nccl_ctx = nccl_ctxs_.at(dev_id);
        auto stream = nccl_ctx.stream();
        auto comm = nccl_ctx.comm_;

        all_reduce_calls.emplace_back([=] {
          PADDLE_ENFORCE(platform::dynload::ncclReduce(
              buffer, recvbuffer, static_cast<size_t>(lod_tensor.numel()),
              static_cast<ncclDataType_t>(dtype), ncclSum, root, comm, stream));
        });
      }

      platform::NCCLGroupGuard guard;
      for (auto &call : all_reduce_calls) {
        call();
      }
    } else {
      PADDLE_THROW("Error");
    }
  }
}
std::string ReduceOpHandle::Name() const { return "reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
