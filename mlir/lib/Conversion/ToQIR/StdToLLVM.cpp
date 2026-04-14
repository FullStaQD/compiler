#include "qcc/Conversion/ToQIR/ToQIR.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/IR/Builders.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

namespace {} // namespace

namespace qcc {

#define GEN_PASS_DEF_STDTOLLVM
#include "qcc/Conversion/ToQIR/ToQIR.h.inc"

struct StdToLLVM final : impl::StdToLLVMBase<StdToLLVM> {
  using StdToLLVMBase::StdToLLVMBase;

protected:
  // FIXME: finish implementation
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    auto* context = funcOp->getContext();
    LLVMConversionTarget target(*context);
    target.addLegalOp<ModuleOp>(); // FIXME: check which are needed
    target.addLegalOp<func::FuncOp>();
    target.addLegalOp<func::ReturnOp>();
    target.addLegalOp<func::CallOp>();

    LLVMTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);

    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace qcc
