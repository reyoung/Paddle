#pragma once
#include <paddle/framework/operator_base.h>
#include <tuple>
#include <type_traits>

namespace paddle {
namespace framework {

struct CPUKernel {
  static bool CanRun(platform::ContextBase* ctx) {
    return dynamic_cast<platform::CpuContext*>(ctx);
  }
};

struct GPUKernel {
  static bool CanRun(platform::ContextBase* ctx) {
    return dynamic_cast<platform::GpuContext*>(ctx);
  }
};

struct NoContextKernel {
  static bool CanRun(platform::ContextBase* ctx) { return ctx == nullptr; }
};

template <typename ParameterType, typename... KERNELS>
struct Operator : public OperatorBase {
  using KERNEL_TUPLE = std::tuple<KERNELS...>;

  void Run(Scope* scope, platform::ContextBase* context) const final {
    std::vector<const Variable*> inputs;
    GetVariables<const Variable*>(&inputs, inputs_, scope);
    std::vector<Variable*> outputs;
    GetVariables<Variable*>(&outputs, outputs_, scope);
    RunImpl<0>()(inputs, outputs, GetParams(), context);
  }

 protected:
  virtual const ParameterType& GetParams() const = 0;

 private:
  template <size_t I, bool atEnd = std::tuple_size<KERNEL_TUPLE>::value == I>
  struct RunImpl {};

  template <size_t I>
  struct RunImpl<I, false> {
    void operator()(const std::vector<const Variable*>& inputs,
                    const std::vector<Variable*>& outputs,
                    const ParameterType& param, platform::ContextBase* ctx) {
      using KERNEL = typename std::tuple_element<I, KERNEL_TUPLE>::type;
      if (KERNEL::CanRun(ctx)) {
        KERNEL::Run(inputs, outputs, param);
      } else {
        RunImpl<I + 1>()(inputs, outputs, param, ctx);
      }
    }
  };

  template <size_t I>
  struct RunImpl<I, true> {
    void operator()(const std::vector<const Variable*>&,
                    const std::vector<Variable*>&, const ParameterType&,
                    platform::ContextBase*) {
      PADDLE_THROW("Not implemented");
    }
  };

  template <typename T>
  void GetVariables(std::vector<T>* vars, const std::vector<std::string>& names,
                    Scope* scope) const {
    vars->reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      vars->push_back(scope->CreateVariable(names[i]));
    }
  }
};

}  // namespace framework
}  // namespace paddle
