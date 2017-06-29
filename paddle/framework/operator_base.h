#pragma once
#include <paddle/framework/attribute_reader.h>
#include <paddle/framework/operator_description.pb.h>
#include <paddle/framework/scope.h>
#include <paddle/platform/context.h>
#include <algorithm>
#include <string>
#include <vector>

namespace paddle {
namespace framework {

struct OperatorBase {
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;

  void Initialize(const OperatorDescription& desc) {
    inputs_.reserve(desc.inputs().size());
    std::copy(desc.inputs().begin(), desc.inputs().end(),
              std::back_inserter(inputs_));
    outputs_.reserve(desc.outputs().size());
    std::copy(desc.outputs().begin(), desc.outputs().end(),
              std::back_inserter(outputs_));
    InitializeAttributes(AttributeReader(desc.attrs()));
  }

  virtual ~OperatorBase() {}  // enable rtti

  virtual void Run(Scope* scope, platform::ContextBase* context) const = 0;

  virtual void InferShape(Scope* scope) const = 0;

 protected:
  virtual void InitializeAttributes(const AttributeReader& reader) = 0;
};
}
}
