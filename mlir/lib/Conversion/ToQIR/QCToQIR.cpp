#include "llvm/IR/Type.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"
#include "qcc/Conversion/ToQIR/constants.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstddef>

using namespace mlir;

namespace {} // namespace

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
