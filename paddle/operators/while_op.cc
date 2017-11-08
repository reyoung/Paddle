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

#include <vector>
#include "paddle/framework/executor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;

constexpr char kStepBlock[] = "step_block";
constexpr char kCondition[] = "Condition";
constexpr char kStepScopes[] = "StepScopes";
class WhileOp : public framework::OperatorBase {
 public:
  WhileOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    //    PADDLE_ENFORCE(...)

    auto &cond = scope.FindVar(Input(kCondition))->Get<framework::LoDTensor>();
    PADDLE_ENFORCE_EQ(cond.dims(), paddle::framework::make_ddim({1}));

    framework::Executor executor(dev_ctx);
    auto *block = Attr<framework::BlockDescBind *>(kStepBlock);
    auto *program = block->Program();

    auto step_scopes = scope.FindVar(kStepScopes)->GetMutable<StepScopeVar>();

    while (cond.data<bool>()[0]) {
      auto &current_scope = scope.NewScope();
      step_scopes->push_back(&current_scope);

      executor.Run(*program, &current_scope, block->ID(),
                   false /*create_local_scope*/);
    }
  }
};

class WhileOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  WhileOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "").AsDuplicable();
    AddInput(kCondition, "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddOutput(kStepScopes, "");
    AddAttr<framework::BlockDescBind *>(kStepBlock,
                                        "The step block inside WhileOp");
    AddComment(R"DOC(
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(while, paddle::operators::WhileOp,
                  paddle::operators::WhileOpMaker);
