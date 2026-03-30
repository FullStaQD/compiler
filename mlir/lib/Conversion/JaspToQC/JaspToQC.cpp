
#include "qcc/Conversion/JaspToQC/JaspToQC.h"

#include "llvm/ADT/Hashing.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
 * The jasp qubit type is mapped to the QC qubit type.
 * Other types like jasp.QubitArray and jasp.QuantumState have
 * no correspondence in QC, and are not converted.
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
 * @brief Converts jasp.get_qubit to qc.alloc
 *
 * @details
 * Allocates a new qubit initialized to the |0⟩ state. Register metadata
 * (name, size, index) is extracted from a preceding `jasp.create_qubits` operation.
 *
 * Both the array size and qubit index values must be created by `arith.const` operations.
 *
 * Example transformation:
 * ```mlir
 * %n = arith.constant dense<7> : tensor<i64>
 * %q_arr, %s1 = jasp.create_qubits %n, %s0 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
 * %i = arith.constant dense<3> : tensor<i64>
 * %q = jasp.get_qubit %q_arr, %i : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
 * // becomes (only the last op is replaced):
 * %q = qc.alloc("qreg", 7, 3) : !qc.qubit
 * ```
 */
struct ConvertJaspGetQubitOp final : OpConversionPattern<jasp::GetQubitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::GetQubitOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter& rewriter) const override {

    auto registerIndex = getConstantIndex(op.getPosition());
    if (!registerIndex)
      return failure();

    auto registerSize = getRegisterSize(op.getQbArray());
    if (!registerSize)
      return failure();

    // Create qc.alloc with preserved register metadata
    rewriter.replaceOpWithNewOp<qc::AllocOp>(op, getQubitArrayName(op.getQbArray()), *registerSize, *registerIndex);

    return success();
  }

private:
  /**
   * @brief Extracts a constant integer attribute from a value.
   *
   * @details
   * Looks for a defining `arith.constant` operation and attempts to extract
   * a single integer value from its attribute. Returns nullopt if the value
   * is not a constant or not a single-element dense integer attribute.
   */
  static std::optional<mlir::IntegerAttr> getConstantIndex(Value value) {
    auto constantOp = value.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constantOp)
      return std::nullopt;

    auto denseAttr = dyn_cast<mlir::DenseIntElementsAttr>(constantOp.getValue());
    if (!denseAttr || denseAttr.getNumElements() != 1)
      return std::nullopt;

    return denseAttr.getValues<mlir::IntegerAttr>()[0];
  }

  /**
   * @brief Extracts the register size from the defining jasp.create_qubits operation.
   *
   * @details
   * Traces the qubit array back to its creation point to determine the total
   * number of qubits allocated in that register.
   */
  static std::optional<mlir::IntegerAttr> getRegisterSize(Value qubitArray) {
    auto createOp = qubitArray.getDefiningOp<jasp::CreateQubitsOp>();
    if (!createOp)
      return std::nullopt;

    return getConstantIndex(createOp.getAmount());
  }

  /**
   * @brief Generates a stable name for a qubit register.
   *
   * @details
   * Creates a unique string identifier based on the opaque pointer of the
   * qubit array value to ensure consistent naming of the register in QC.
   */
  mlir::StringAttr getQubitArrayName(Value qubitArray) const {
    size_t hash = llvm::hash_value(qubitArray.getAsOpaquePointer());
    return mlir::StringAttr::get(getContext(), "qreg_" + llvm::utohexstr(hash));
  }
};

/**
 * @brief Converts jasp.consume_quantum_kernel op to arith.constant
 *
 * @details
 * We do not lower the functionality of returning a success value
 * for the kernel execution. A constant value of 1 is returned,
 * assuming a successful execution.
 *
 * Example transformation:
 * ```mlir
 * %success = jasp.consume_quantum_kernel %state : !jasp.QuantumState -> tensor<i1>
 * // becomes (only the last op is replaced):
 * %success = arith.constant dense<1> : tensor<i1>
 * ```
 */
struct ConvertJaspConsumeQuantumKernelOp final : OpConversionPattern<jasp::ConsumeQuantumKernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::ConsumeQuantumKernelOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter& rewriter) const override {

    auto type = op.getType();
    auto trueAttr = rewriter.getBoolAttr(true);
    auto denseAttr = DenseElementsAttr::get(type, trueAttr);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, type, denseAttr);

    return success();
  }
};

/**
 * @brief Converts a jasp gate to QC
 *
 * @details
 * TODO
 *
 * Example:
 * ```mlir
 * %state_out = jasp.quantum_gate "x" (%q), %state_in : (!jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
 * ```
 * is converted to
 * ```mlir
 * qc.x %q : !qc.qubit
 * ```
 */
struct ConvertJaspQuantumGateOp final : OpConversionPattern<jasp::QuantumGateOp> {
  using OpConversionPattern<jasp::QuantumGateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::QuantumGateOp op, jasp::QuantumGateOp::Adaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {

    auto gate_name = op.getGateType();
    if (gate_name != "h")
      return failure();

    auto qcQubit = adaptor.getGateOperands().front();

    qc::HOp::create(rewriter, op.getLoc(), qcQubit);
    rewriter.replaceOp(op, qcQubit);

    return success();
  }
};

/**
 * @brief Converts jasp.measure to qc.measure
 *
 * @details
 * The conversion is straightforward. However, the measurement result in jasp is
 * of type `tensor<i1>`, whereas in QC it is of type `i1`.
 * Therefore, a conversion between these types is inserted.
 *
 * Example transformation:
 * ```mlir
 * %measured, %state1 = jasp.measure %q, %state0 : !jasp.Qubit, !jasp.QuantumState -> tensor<i1>, !jasp.QuantumState
 * // becomes:
 * %c = qc.measure %q : !qc.qubit -> i1
 * %measured = tensor.from_elements %c : tensor<i1>
 * ```
 */
struct ConvertJaspMeasureOp final : OpConversionPattern<jasp::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::MeasureOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto qcQubit = adaptor.getMeasQ();

    // Create qc.measure (in-place operation, returns only bit)
    // Preserve register metadata for output recording
    auto qcMeasureOp = qc::MeasureOp::create(rewriter, op.getLoc(), qcQubit);

    auto measureBit = qcMeasureOp.getResult();

    // Create tensor from the i1 result to match jasp.measure's return type
    auto tensorResult = tensor::FromElementsOp::create(rewriter, op.getLoc(), op.getType(1), measureBit);

    rewriter.replaceOp(op, {qcQubit, tensorResult.getResult()});

    return success();
  }
};

/**
 * @brief Converts jasp.delete_qubits to qc.dealloc
 *
 * @details
 * TODO
 *
 * Example transformation:
 * ```mlir
 * %state1 = jasp.delete_qubits %qubit_array, %state0 : !jasp.QubitArray, !jasp.QuantumState -> !jasp.QuantumState
 * // becomes:
 * qc.dealloc %q_0 : !qc.qubit  // for every qubit from the array
 * ```
 */
struct ConvertJaspDeallocOp final : OpConversionPattern<jasp::DeleteQubitsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::DeleteQubitsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto array = op.getQubits();

    // TODO: improve
    rewriter.eraseOp(op);

    return success();
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

    target.addIllegalDialect<JaspDialect>();
    target.addLegalDialect<QCDialect>();

    // Register operation conversion patterns
    patterns.add<ConvertJaspGetQubitOp, ConvertJaspConsumeQuantumKernelOp, ConvertJaspQuantumGateOp,
                 ConvertJaspMeasureOp, ConvertJaspDeallocOp>(typeConverter, context);

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
