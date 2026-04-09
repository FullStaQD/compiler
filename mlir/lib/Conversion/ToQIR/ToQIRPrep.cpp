#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "qcc/Conversion/ToQIR/Constants.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"

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
    auto* ctx = moduleOp.getContext();
    OpBuilder builder(ctx);

    // Runtime functions:
    createVoidFnDecl(qcc::qirRtInit, 1);
    createRtReadResultDecl();

    // QIS:
    auto fnMZ = createVoidFnDecl(qcc::qirQisMZ, 2);
    fnMZ.setArgAttr(1, "llvm.writeonly", builder.getUnitAttr());
    fnMZ->setAttr("passthrough", builder.getStrArrayAttr({"irreversible"}));
    createVoidFnDecl(qcc::qirQisH, 1);
    createVoidFnDecl(qcc::qirQisX, 1);
    createVoidFnDecl(qcc::qirQisCX, 2);

    addQIRModuleFlags();
  }

private:
  /// Insert `llvm.func` with signature `fnName(ptr, ptr, ...) -> void`.
  LLVM::LLVMFuncOp createVoidFnDecl(StringRef fnName, int numPtrs) {
    ModuleOp moduleOp = getOperation();
    auto* context = moduleOp.getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto ptrType = LLVM::LLVMPointerType::get(context);
    SmallVector<Type, 2> argTypes(numPtrs, ptrType);
    auto fnType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), argTypes);

    auto fnDecl = LLVM::LLVMFuncOp::create(builder, moduleOp.getLoc(), fnName, fnType);

    return fnDecl;
  }

  /// Insert `llvm.func` with signature `__quantum__rt__read_result(ptr readonly) -> void`.
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

  /// Create the module flags which specify the capabilities which the backend needs to support.
  ///
  /// Of course we in turn also have to ensure that our output does not go beyond those capabilities.
  ///
  /// TODO: Currently we hardcode the capabilities. In the future we have to query those (e.g. QDMI).
  void addQIRModuleFlags() {
    ModuleOp module = getOperation();
    auto* ctx = module.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());
    auto loc = module.getLoc();

    auto createFlag = [&](LLVM::ModFlagBehavior behavior, StringRef name, int32_t val) {
      // LLVM seems to normalize to i32 values. Even if we would use e.g. a i1
      // attribute for the boolean flags `mlir-translate` would still force it
      // to i32. Hence we stick to i32 always here.
      return LLVM::ModuleFlagAttr::get(ctx, behavior, builder.getStringAttr(name), builder.getI32IntegerAttr(val));
    };

    // NOTE: Missing flags are implicitly set to "false".
    SmallVector<Attribute> flags = {
        createFlag(LLVM::ModFlagBehavior::Error, "qir_major_version", 2),
        createFlag(LLVM::ModFlagBehavior::Max, "qir_minor_version", 0),
        createFlag(LLVM::ModFlagBehavior::Error, "dynamic_qubit_management", 0),
        createFlag(LLVM::ModFlagBehavior::Error, "dynamic_result_management", 0),
        createFlag(LLVM::ModFlagBehavior::Error, "ir_functions", 1),
        // backwards_branching: 0: no, 1: only "unrollable" loops, 2: conditionally terminating loops
        createFlag(LLVM::ModFlagBehavior::Error, "backwards_branching", 1),
        createFlag(LLVM::ModFlagBehavior::Error, "multiple_target_branching", 0),
        createFlag(LLVM::ModFlagBehavior::Error, "multiple_return_points", 0)};

    LLVM::ModuleFlagsOp::create(builder, loc, builder.getArrayAttr(flags));
  }
};

} // namespace qcc
