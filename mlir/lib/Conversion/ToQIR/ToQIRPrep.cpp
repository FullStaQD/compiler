#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "qcc/Conversion/ToQIR/Constants.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>

using namespace mlir;

namespace qcc {

#define GEN_PASS_DEF_TOQIRPREP
#include "qcc/Conversion/ToQIR/ToQIR.h.inc"

struct ToQIRPrep final : public impl::ToQIRPrepBase<ToQIRPrep> {
  using impl::ToQIRPrepBase<ToQIRPrep>::ToQIRPrepBase;

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto* context = moduleOp.getContext();

    // Runtime functions:
    createFnDecl(qcc::qirRtInit, 1);
    createRtReadResultDecl();

    // QIS:
    createFnDecl(qcc::qirQisMZ, 2, true); // FIXME: second arg is actually writeonly
    createFnDecl(qcc::qirQisH, 1);
    createFnDecl(qcc::qirQisX, 1);
    createFnDecl(qcc::qirQisCX, 2);

    addQIRModuleFlags();
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

  void createRtReadResultDecl() {
    ModuleOp moduleOp = getOperation();
    auto* ctx = moduleOp.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto i1Type = IntegerType::get(ctx, 1);
    auto fnType = LLVM::LLVMFunctionType::get(i1Type, {ptrType});

    auto fnDecl = LLVM::LLVMFuncOp::create(builder, moduleOp.getLoc(), qcc::qirRtReadResult, fnType);
    fnDecl.setArgAttr(0, "llvm.readonly", builder.getUnitAttr());
  }

  /// FIXME: docstring
  void addQIRModuleFlags() {
    ModuleOp module = getOperation();
    auto* ctx = module.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());
    auto loc = module.getLoc();

    auto createFlag = [&](LLVM::ModFlagBehavior behavior, StringRef name, int32_t val) {
      return LLVM::ModuleFlagAttr::get(ctx, behavior, builder.getStringAttr(name), builder.getI32IntegerAttr(val));
    };

    SmallVector<Attribute> flags = {createFlag(LLVM::ModFlagBehavior::Error, "qir_major_version", 2),
                                    createFlag(LLVM::ModFlagBehavior::Max, "qir_minor_version", 0),
                                    createFlag(LLVM::ModFlagBehavior::Error, "dynamic_qubit_management", 0),
                                    createFlag(LLVM::ModFlagBehavior::Error, "dynamic_result_management", 0),
                                    createFlag(LLVM::ModFlagBehavior::Error, "ir_functions", 0),
                                    createFlag(LLVM::ModFlagBehavior::Error, "backwards_branching", 0),
                                    createFlag(LLVM::ModFlagBehavior::Error, "multiple_target_branching", 0),
                                    createFlag(LLVM::ModFlagBehavior::Error, "multiple_return_points", 0)};

    LLVM::ModuleFlagsOp::create(builder, loc, builder.getArrayAttr(flags));
  }
};

} // namespace qcc
