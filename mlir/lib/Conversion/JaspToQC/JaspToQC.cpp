
#include "qcc/Conversion/JaspToQC/JaspToQC.h"

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "qcc/Dialect/Jasp/IR/Jasp.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <utility>

using namespace jasp;
namespace qcc {
using namespace mlir;
using namespace mlir::qc;

#define GEN_PASS_DEF_JASPTOQC
#include "qcc/Conversion/JaspToQC/JaspToQC.h.inc"

/**
 * @brief Type converter for jasp-to-QC conversion
 *
 * @details
 * Handles type conversion between the jasp and QC dialects.
 * TODO.
 */
class JaspToQCTypeConverter final : public TypeConverter {
public:
  explicit JaspToQCTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    // Convert jasp qubit references to QC qubit references
    addConversion([ctx](jasp::QubitType /*type*/) -> Type { return qc::QubitType::get(ctx); });
  }
};

/**
 * @brief Pass implementation for jasp-to-QC conversion
 *
 * @details
 * TODO
 */
struct JaspToQC final : impl::JaspToQCBase<JaspToQC> {
  using JaspToQCBase::JaspToQCBase;

protected:
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    JaspToQCTypeConverter typeConverter(context);

    // Configure conversion target: jasp illegal, QC legal
    target.addIllegalDialect<JaspDialect>();
    target.addLegalDialect<QCDialect>();

    // Register operation conversion patterns
    // Note: No state tracking needed - OpAdaptors handle type conversion
    // patterns.add<>(typeConverter, context);

    // Conversion of jasp types in func.func signatures
    // Note: This currently has limitations with signature changes
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) && typeConverter.isLegal(&op.getBody());
    });

    // Conversion of jasp types in func.return
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>([&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

    // Conversion of jasp types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>([&](const func::CallOp op) { return typeConverter.isLegal(op); });

    // Conversion of jasp types in control-flow ops (e.g., cf.br, cf.cond_br)
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace qcc
