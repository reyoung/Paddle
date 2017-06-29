#include <gtest/gtest.h>
#include <paddle/framework/operator_base.h>

TEST(OperatorBase, all) {
  using paddle::framework::OperatorBase;
  using paddle::framework::OperatorDescription;
  using paddle::framework::AttributeReader;
  using paddle::framework::Scope;
  using paddle::platform::ContextBase;
  struct TestOpImpl : public OperatorBase {
    float scale_{1.0};

    void Run(Scope* scope, ContextBase* ctx) const override {
      PADDLE_THROW("Not implemented");
    }

    void InferShape(Scope* scope) const override {
      PADDLE_THROW("Not implemented");
    }

   protected:
    void InitializeAttributes(const AttributeReader& reader) override {
      if (reader.Contains<float>("scale")) {
        scale_ = reader.Get<float>("scale");
        PADDLE_ENFORCE(
            scale_ > 0.0f,
            "Attribute scale must larger than 0.0, but actual is %.2f", scale_);
      }
    }
  };

  OperatorDescription desc;
  desc.set_type("test_op");
  *desc.mutable_inputs()->Add() = "input1";
  *desc.mutable_inputs()->Add() = "input2";
  *desc.mutable_outputs()->Add() = "output";
  (*desc.mutable_attrs())["scale"].set_f(3.0f);

  TestOpImpl op1;
  op1.Initialize(desc);
  ASSERT_EQ(2UL, op1.inputs_.size());
  ASSERT_EQ(1UL, op1.outputs_.size());
  ASSERT_NEAR(3.0f, op1.scale_, 1e-5);

  TestOpImpl op2;
  (*desc.mutable_attrs())["scale"].set_f(-1.0);
  ASSERT_THROW(op2.Initialize(desc), paddle::framework::EnforceNotMet);
}