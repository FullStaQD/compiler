#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"
#include "qcc/Conversion/ToQIR/constants.h"

using namespace mlir;

namespace qcc {

#define GEN_PASS_DEF_TOQIRPREP
#include "qcc/Conversion/ToQIR/ToQIR.h.inc"

struct ToQIRPrep final : public impl::ToQIRPrepBase<ToQIRPrep> {
  using impl::ToQIRPrepBase<ToQIRPrep>::ToQIRPrepBase;

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto context = moduleOp.getContext();

    // Runtime functions:
    createFnDecl(qcc::QIR_RT_INIT, 1);

    // QIS:
    createFnDecl(qcc::QIR_QIS_MZ, 2, true);
    createFnDecl(qcc::QIR_QIS_H, 1);
    createFnDecl(qcc::QIR_QIS_X, 1);
  }

private:
  /// FIXME: docstring
  void createFnDecl(StringRef fnName, int numPtrs, bool irreversible = false) {
    ModuleOp moduleOp = getOperation();
    auto* context = moduleOp.getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    // Prepare signature: (ptr, ptr, ...) -> void
    auto ptrType = LLVM::LLVMPointerType::get(context);
    SmallVector<Type, 2> argTypes(numPtrs, ptrType);
    auto fnType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), argTypes);

    auto fnDecl = LLVM::LLVMFuncOp::create(builder, moduleOp.getLoc(), fnName, fnType);

    if (irreversible) {
      fnDecl->setAttr("passthrough", builder.getStrArrayAttr({"irreversible"}));
    }
  }
};

} // namespace qcc
