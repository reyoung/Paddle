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

#include "paddle/fluid/framework/details/multi_devices_graph_builder.h"
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include "paddle/fluid/framework/details/send_op_handle.h"
#include "paddle/fluid/framework/scope.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/details/nccl_all_reduce_op_handle.h"
#endif

#include <string>
#include <vector>

namespace paddle {
namespace framework {
namespace details {

#ifdef PADDLE_WITH_CUDA
MultiDevSSAGraphBuilder::MultiDevSSAGraphBuilder(
    const std::vector<platform::Place> &places,
    const std::string &loss_var_name,
    const std::unordered_set<std::string> &params,
    const std::vector<Scope *> &local_scopes,
    platform::NCCLContextMap *nccl_ctxs)
    : loss_var_name_(loss_var_name),
      places_(places),
      local_scopes_(local_scopes),
      nccl_ctxs_(nccl_ctxs) {
#else
MultiDevSSAGraphBuilder::MultiDevSSAGraphBuilder(
    const std::vector<platform::Place> &places,
    const std::string &loss_var_name,
    const std::unordered_set<std::string> &params,
    const std::vector<Scope *> &local_scopes)
    : loss_var_name_(loss_var_name),
      places_(places),
      local_scopes_(local_scopes) {
#endif
  for (auto &p : params) {
    grad_names_.insert(GradVarName(p));
  }
}

void MultiDevSSAGraphBuilder::CreateOpHandleIOs(SSAGraph *result,
                                                const OpDesc &op,
                                                const platform::Place &p,
                                                const size_t &i) const {
  auto *op_handle = result->ops_.back().get();
  op_handle->dev_ctxes_[p] = platform::DeviceContextPool::Instance().Get(p);

  auto var_names = op.InputArgumentNames();

  for (auto &each_var_name : var_names) {
    VarHandle *var = CreateOrGetLatestVarHandle(result, each_var_name, p, i);
    op_handle->AddInput(var);
  }

  var_names = op.OutputArgumentNames();

  for (auto &each_var_name : var_names) {
    CreateOpOutput(result, op_handle, each_var_name, p, i);
  }
}

std::unique_ptr<SSAGraph> MultiDevSSAGraphBuilder::Build(
    const ProgramDesc &program) const {
  auto graph = new SSAGraph();
  SSAGraph &result = *graph;
  std::unordered_set<std::string> og_has_been_broadcast;

  // We cannot invoke resize. It is a bug of GCC 4.8
  result.vars_ = std::vector<
      std::unordered_map<std::string, std::vector<std::unique_ptr<VarHandle>>>>(
      places_.size());

  bool is_forwarding = true;

  std::vector<std::unordered_set<std::string>> grad_deps_sets;
  std::vector<int> devices_;
  std::unordered_set<std::string> reduced_grad;

  for (auto *op : program.Block(0).AllOps()) {
    if (is_forwarding) {
      if (AppendForwardOp(&result, op)) {
        is_forwarding = false;
      }
    } else {
      // FIXME(yy): Do not hard code like this
      if (IsScaleGradOp(op)) {
        continue;  // Drop fill 1. for backward coeff;
      } else if (IsSendOp(op)) {
        // append send op if program is distributed trainer main program.
        // always use the first device

        auto &p = places_[0];
        auto *s = local_scopes_[0];
        // FIXME(wuyi): send op always copy from GPU 0
        result.ops_.emplace_back(new SendOpHandle(*op, s, p));
        // Create inputs for output on original place and no ssa output
        // is created for send op.
        CreateOpHandleIOs(&result, *op, p, 0);
        continue;
      } else {
        std::vector<std::string> g_names = GetParamGradientNames(op);
        bool need_reduce = !g_names.empty();
        for (auto &g_name : g_names) {
          if (!need_reduce) break;
          need_reduce = !reduced_grad.count(g_name);
        }

        if (need_reduce) {
          for (size_t i = 0; i < places_.size(); ++i) {
            auto &p = places_[i];
            auto *s = local_scopes_[i];
            result.ops_.emplace_back(new ComputationOpHandle(*op, s, p));
            CreateOpHandleIOs(&result, *op, p, i);
          }

          for (auto &g_name : g_names) {
            devices_.emplace_back(devices_.size() % places_.size());
            int dev_id = devices_.back();
            grad_deps_sets.emplace_back({g_name});
          }
        }
      }
    }
  }

  /*
    Dependency graph has been constructed. However, there are still data
    harzaeds need to be handled.
   */
  PolishGraphToSupportDataHazards(&result);

  /*
   * Only variables should be the leaves of graph.
   */
  AddOutputToLeafOps(&result);

  if (VLOG_IS_ON(10)) {
    std::ostringstream sout;
    PrintGraphviz(*graph, sout);
    VLOG(10) << sout.str();
  }

  return std::unique_ptr<SSAGraph>(graph);
}

std::vector<std::string> MultiDevSSAGraphBuilder::GetParamGradientNames(
    const OpDesc *op) const {
  auto var_names = op->OutputArgumentNames();
  std::vector<std::__cxx11::string> g_names;
  for (auto &var_name : var_names) {
    if (grad_names_.count(var_name)) {
      g_names.emplace_back(var_name);
    }
  }
  return g_names;
}

bool MultiDevSSAGraphBuilder::IsSendOp(const OpDesc *op) const {
  return op->Type() == "send";
}

bool MultiDevSSAGraphBuilder::IsScaleGradOp(const OpDesc *op) const {
  return op->OutputArgumentNames().size() == 1 &&
         op->OutputArgumentNames()[0] == GradVarName(loss_var_name_);
}

bool MultiDevSSAGraphBuilder::AppendForwardOp(SSAGraph *result_ptr,
                                              const OpDesc *op) const {
  auto &result = *result_ptr;
  bool res = false;
  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    auto *s = local_scopes_[i];

    result.ops_.emplace_back(new ComputationOpHandle(*op, s, p));
    auto *op_handle = result.ops_.back().get();
    this->CreateOpHandleIOs(&result, *op, p, i);

    auto var_names = op->OutputArgumentNames();

    if (var_names.size() == 1 && var_names[0] == loss_var_name_) {
// Insert ScaleCost OpHandle
#ifdef PADDLE_WITH_CUDA
      auto *communication_dev_ctx = nccl_ctxs_->DevCtx(p);
#else
      auto *communication_dev_ctx =
          platform::DeviceContextPool::Instance().Get(platform::CPUPlace());
#endif

      op_handle = new ScaleLossGradOpHandle(local_scopes_.size(), s, p,
                                            communication_dev_ctx);
      result.ops_.emplace_back(op_handle);

      // FIXME: Currently ScaleLossGradOp only use device_count as scale
      // factor. So it does not depend on any other operators.
      // VarHandle *loss = GetVarHandle(loss_var_name, place);
      // loss->pending_ops_.emplace_back(op_handle);
      // op_handle->inputs_.emplace_back(loss);

      CreateOpOutput(&result, op_handle, GradVarName(loss_var_name_), p, i);
      res = true;
    }
  }
  return res;
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
