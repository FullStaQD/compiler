#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <llvm/Support/raw_ostream.h>

namespace qcc {

#define GEN_PASS_DEF_QCTOQIRADAPTIVE
#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h.inc"

struct QCToQIRAdaptive final : impl::QCToQIRAdaptiveBase<QCToQIRAdaptive> {
  using QCToQIRAdaptiveBase::QCToQIRAdaptiveBase;

protected:
  void runOnOperation() override {
    // FIXME: implement this!
  }
};

} // namespace qcc
