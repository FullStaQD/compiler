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

    createQisPtrPtr(qcc::QIR_QIS_MZ, true);
    createQisPtr(qcc::QIR_QIS_X, false);
    createQisPtr(qcc::QIR_QIS_H, false);
  }

private:
  void createQisPtr(StringRef fnName, bool irreversible) {
    ModuleOp moduleOp = getOperation();
    auto context = moduleOp.getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    const auto ptrType = LLVM::LLVMPointerType::get(context);
    auto fnType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), {ptrType});

    auto fnDecl = LLVM::LLVMFuncOp::create(builder, moduleOp->getLoc(), fnName, fnType);

    if (irreversible) {
      fnDecl->setAttr("passthrough", builder.getStrArrayAttr({"irreversible"}));
    }
  }

  void createQisPtrPtr(StringRef fnName, bool irreversible) {
    ModuleOp moduleOp = getOperation();
    auto context = moduleOp.getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    const auto ptrType = LLVM::LLVMPointerType::get(context);
    auto fnType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), {ptrType, ptrType});

    auto fnDecl = LLVM::LLVMFuncOp::create(builder, moduleOp->getLoc(), fnName, fnType);

    if (irreversible) {
      fnDecl->setAttr("passthrough", builder.getStrArrayAttr({"irreversible"}));
    }
  }
};

} // namespace qcc
