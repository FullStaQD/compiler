
#include "qcc/Conversion/JaspToQC/JaspToQC.h"

#include "llvm/ADT/Hashing.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

    addConversion([ctx](jasp::QubitType /*type*/) -> Type { return qc::QubitType::get(ctx); });
    addConversion([ctx](jasp::QubitArrayType /*type*/) -> Type {
      return MemRefType::get({ShapedType::kDynamic}, qc::QubitType::get(ctx));
    });
  }
};

/**
 * @brief Converts jasp.create_quantum_kernel by deleting it
 *
 * @details
 * The jasp.QuantumState is used for tracking state in the jasp dialect.
 * Since QC dialect operations are side-effecting, the initial state creation is dropped.
 *
 * Note: The IR must be ordered correctly at the start of this pass.
 */
struct ConvertJaspCreateQuantumKernelOp final : OpConversionPattern<jasp::CreateQuantumKernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::CreateQuantumKernelOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts jasp.create_qubits to memref.alloc
 *
 * @details
 * The jasp.QuantumState is dropped. The size tensor is extracted to an index
 * to allocate a memref of qubits.
 *
 * Example transformation:
 * ```mlir
 * %q_arr, %state1 = jasp.create_qubits %tensor_index, %state0 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray,
 * !jasp.QuantumState
 * // becomes:
 * %index = tensor.extract %tensor_index[] : tensor<i64>
 * %q_arr = memref.alloc(%index) : memref<?x!qc.qubit>
 * ```
 */
struct ConvertJaspCreateQubitsOp final : OpConversionPattern<jasp::CreateQubitsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::CreateQubitsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto index = tensor::ExtractOp::create(rewriter, loc, adaptor.getAmount(), ValueRange{});
    auto memrefType = getTypeConverter()->convertType(op.getType(0));
    auto alloc = memref::AllocOp::create(rewriter, loc, cast<MemRefType>(memrefType), ValueRange{index});
    rewriter.replaceOp(op, {alloc.getResult(), nullptr});
    return success();
  }
};

/**
 * @brief Converts jasp.get_qubit to memref.load
 *
 * @details
 * The conversion is straightforward. Only the index type needs to be
 * converted.
 *
 * Example transformation:
 * ```mlir
 * %q = jasp.get_qubit %q_arr, %i : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
 * // becomes:
 * %j = tensor.extract %i[] : tensor<i64>
 *  %q = memref.load %q_arr[%j] : memref<?x!qc.qubit>
 * ```
 */
struct ConvertJaspGetQubitOp final : OpConversionPattern<jasp::GetQubitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::GetQubitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto indexTensor = adaptor.getPosition();

    auto extractOp = tensor::ExtractOp::create(rewriter, loc, indexTensor, ValueRange{});
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getQbArray(), ValueRange{extractOp});

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
    // TODO: Implement gates
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
 * memref.dealloc %qubit_array : memref<?x!qc.qubit>
 * ```
 */
struct ConvertJaspDeleteQubitsOp final : OpConversionPattern<jasp::DeleteQubitsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::DeleteQubitsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto array = adaptor.getQubits();
    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, array);
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
    patterns.add<ConvertJaspCreateQuantumKernelOp, ConvertJaspCreateQubitsOp, ConvertJaspGetQubitOp,
                 ConvertJaspConsumeQuantumKernelOp, ConvertJaspQuantumGateOp, ConvertJaspMeasureOp,
                 ConvertJaspDeleteQubitsOp>(typeConverter, context);

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
