#pragma once

namespace paddle {
namespace platform {
struct ContextBase {
  virtual ~ContextBase() {}
};

struct CpuContext : public ContextBase {};

struct GpuContext : public ContextBase {};
}
}
