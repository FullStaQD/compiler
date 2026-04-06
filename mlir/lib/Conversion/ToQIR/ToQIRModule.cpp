#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"

#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace qcc {

#define GEN_PASS_DEF_TOQIRMODULE
#include "qcc/Conversion/ToQIR/ToQIR.h.inc"

struct ToQIRModule : impl::ToQIRModuleBase<ToQIRModule> {
  using ToQIRModuleBase::ToQIRModuleBase;

protected:
  void runOnOperation() override {
    // FIXME: finish impl

    ModuleOp moduleOp = getOperation();
    MLIRContext* context = moduleOp.getContext();

    LLVMConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    LLVMTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);

    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns); // FIXME: maybe not needed here
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace qcc
