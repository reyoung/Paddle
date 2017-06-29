#include <gtest/gtest.h>
#include <paddle/framework/operator.h>
using paddle::framework::Operator;
using paddle::framework::Scope;
using paddle::framework::AttributeReader;
using paddle::framework::CPUKernel;
using paddle::framework::Variable;
using paddle::framework::OperatorDescription;
using paddle::platform::CpuContext;
using paddle::platform::GpuContext;

struct TestOpParam {
  float scale_{1.0f};
};

template <typename... KERNELS>
struct TestOp : public Operator<TestOpParam, KERNELS...> {
  TestOpParam param_;

  void InferShape(Scope* scope) const final { PADDLE_THROW("Unimplemented"); }

 protected:
  void InitializeAttributes(const AttributeReader& reader) final {
    if (reader.Contains<float>("scale")) {
      param_.scale_ = reader.Get<float>("scale");
    }
  }

  const TestOpParam& GetParams() const final { return param_; }
};

struct TestOpCpuKernel : public CPUKernel {
  static void Run(const std::vector<const Variable*>& inputs,
                  const std::vector<Variable*>& outputs,
                  const TestOpParam& param) {}
};

TEST(Operator, all) {
  TestOp<TestOpCpuKernel> op;
  OperatorDescription desc;
  *desc.mutable_inputs()->Add() = "input1";
  *desc.mutable_inputs()->Add() = "input2";
  *desc.mutable_outputs()->Add() = "output";
  op.Initialize(desc);
  Scope scope;
  CpuContext ctx;
  op.Run(&scope, &ctx);

  GpuContext gpuCtx;
  ASSERT_THROW(op.Run(&scope, &gpuCtx), paddle::framework::EnforceNotMet);
}