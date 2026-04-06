#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"

#include <mlir/Pass/Pass.h>

namespace qcc {

#define GEN_PASS_DEF_QCTOQIR
#include "qcc/Conversion/ToQIR/ToQIR.h.inc"

struct QCToQIR final : impl::QCToQIRBase<QCToQIR> {
  using QCToQIRBase::QCToQIRBase;

protected:
  void runOnOperation() override {
    // FIXME: implement conversion from QC dialect to QIR
  }
};

} // namespace qcc
