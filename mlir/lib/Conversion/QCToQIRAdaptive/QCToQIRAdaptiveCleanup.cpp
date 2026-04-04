#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h"

#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace qcc {

#define GEN_PASS_DEF_QCTOQIRADAPTIVECLEANUP
#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h.inc"

struct QCToQIRAdaptiveCleanup : impl::QCToQIRAdaptiveCleanupBase<QCToQIRAdaptiveCleanup> {
  using QCToQIRAdaptiveCleanupBase::QCToQIRAdaptiveCleanupBase;

protected:
  void runOnOperation() override {
    llvm::errs() << "!!! cleanup !!!\n";

    ModuleOp moduleOp = getOperation();
  }
};

} // namespace qcc
