
#include "qcc/Conversion/JaspToQC/JaspToQC.h"

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

#include <cassert>
#include <cstddef>
#include <utility>

namespace qcc {
using namespace jasp;
using namespace mlir;
using namespace mlir::qc;

#define GEN_PASS_DEF_JASPTOQC
#include "qcc/Conversion/JaspToQC/JaspToQC.h.inc"

/**
 * @brief Type converter for jasp-to-QC conversion
 *
 * @details
 * Handles type conversion between the jasp and QC dialects.
 *  - The jasp qubit type is mapped to the QC qubit type.
 *  - !jasp.QubitArray is mapped to memref<?x!qc.qubit>.
 *  - TODO: !jasp.QuantumState is destroyed.
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

// TODO: add operation support: parity, barrier, slice, fuse, get_size, reset

/**
 * @brief Converts jasp.create_quantum_kernel by deleting it
 *
 * @details
 * The jasp.QuantumState is used for tracking state in the jasp dialect.
 * Since QC dialect operations are side-effecting, the initial state creation is dropped.
 *
 * NOTE: For this to work, the IR must be ordered correctly at the start of this pass.
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
 * %index_i64 = tensor.extract %tensor_index[] : tensor<i64>
 * %index = arith.index_cast %index_i64 : i64 to index
 * %q_arr = memref.alloc(%index) : memref<?x!qc.qubit>
 * ```
 */
struct ConvertJaspCreateQubitsOp final : OpConversionPattern<jasp::CreateQubitsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::CreateQubitsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto extracted = tensor::ExtractOp::create(rewriter, loc, adaptor.getAmount(), ValueRange{});
    auto index = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), extracted);
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, qc::QubitType::get(getContext()));
    auto alloc = memref::AllocOp::create(rewriter, loc, cast<MemRefType>(memrefType), ValueRange{index});
    rewriter.replaceOpWithMultiple(op, {alloc.getResult(), ValueRange()});
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
 * %i_i64 = tensor.extract %i[] : tensor<i64>
 * %i_index = arith.index_cast %i_i64 : i64 to index
 *  %q = memref.load %q_arr[%i_index] : memref<?x!qc.qubit>
 * ```
 */
struct ConvertJaspGetQubitOp final : OpConversionPattern<jasp::GetQubitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(jasp::GetQubitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto indexTensor = adaptor.getPosition();

    auto extractOp = tensor::ExtractOp::create(rewriter, loc, indexTensor, ValueRange{});
    auto index = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), extractOp);
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getQbArray(), ValueRange{index});

    return success();
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
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, type, cast<TypedAttr>(denseAttr));

    return success();
  }
};

/**
 * @brief Converts a jasp gate to QC
 *
 * @details
 * Converts generic jasp.quantum_gate operations into specific QC dialect
 * operations based on the gate name string.
 *
 * This pattern handles the extraction of parameters from tensors, the
 * mapping of target qubits, and the conversion of controlled gates into
 * `qc.ctrl` operations containing the base gate in their region.
 *
 * Example:
 * ```mlir
 * %state_out = jasp.quantum_gate "h" (%q), %state_in : (!jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
 * ```
 * is converted to
 * ```mlir
 * qc.h %q : !qc.qubit
 * ```
 */
struct ConvertJaspQuantumGateOp final : OpConversionPattern<jasp::QuantumGateOp> {
  using OpConversionPattern<jasp::QuantumGateOp>::OpConversionPattern;

/**
 * @brief Macro to attempt converting a jasp gate to a specific QC operation
 *
 * @param NAME The string identifier of the gate in jasp
 * @param OPTYPE The MLIR operation type in the QC dialect
 * @param N_CONTROLS Number of control qubits
 * @param N_TARGETS Number of target qubits
 * @param N_PARAMS Number of floating point parameters
 */
#define TRY_CONVERT_GATE(NAME, OPTYPE, N_CONTROLS, N_TARGETS, N_PARAMS)                                                \
  if (gateName == NAME) {                                                                                              \
    return convertGate<OPTYPE>(op, adaptor, rewriter, std::make_index_sequence<N_CONTROLS>{},                          \
                               std::make_index_sequence<N_TARGETS>{}, std::make_index_sequence<N_PARAMS>{});           \
  }

  LogicalResult matchAndRewrite(jasp::QuantumGateOp op, jasp::QuantumGateOp::Adaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {

    auto gateName = op.getGateType();

    TRY_CONVERT_GATE("id", qc::IdOp, 0, 1, 0);
    TRY_CONVERT_GATE("gphase", qc::GPhaseOp, 0, 0, 1);
    TRY_CONVERT_GATE("x", qc::XOp, 0, 1, 0);
    TRY_CONVERT_GATE("y", qc::YOp, 0, 1, 0);
    TRY_CONVERT_GATE("z", qc::ZOp, 0, 1, 0);
    TRY_CONVERT_GATE("h", qc::HOp, 0, 1, 0);
    TRY_CONVERT_GATE("cx", qc::XOp, 1, 1, 0);
    TRY_CONVERT_GATE("cy", qc::YOp, 1, 1, 0);
    TRY_CONVERT_GATE("cz", qc::ZOp, 1, 1, 0);
    TRY_CONVERT_GATE("p", qc::POp, 0, 1, 1);
    TRY_CONVERT_GATE("cp", qc::POp, 1, 1, 1);
    TRY_CONVERT_GATE("rx", qc::RXOp, 0, 1, 1);
    TRY_CONVERT_GATE("ry", qc::RYOp, 0, 1, 1);
    TRY_CONVERT_GATE("rz", qc::RZOp, 0, 1, 1);
    TRY_CONVERT_GATE("crz", qc::RZOp, 1, 1, 1);
    TRY_CONVERT_GATE("s", qc::SOp, 0, 1, 0);
    TRY_CONVERT_GATE("t", qc::TOp, 0, 1, 0);
    TRY_CONVERT_GATE("sx", qc::SXOp, 0, 1, 0);
    TRY_CONVERT_GATE("swap", qc::SWAPOp, 0, 2, 0);
    TRY_CONVERT_GATE("rxx", qc::RXXOp, 0, 2, 1);
    TRY_CONVERT_GATE("rzz", qc::RZZOp, 0, 2, 1);
    TRY_CONVERT_GATE("xxyy", qc::XXPlusYYOp, 0, 2, 2);
    TRY_CONVERT_GATE("u3", qc::UOp, 0, 1, 3);

    // TODO: add multi-controlled gates

    return failure();
  }

#undef TRY_CONVERT_GATE

private:
  inline Value extractParam(jasp::QuantumGateOp op, Value paramTensor, ConversionPatternRewriter& rewriter) const {
    return tensor::ExtractOp::create(rewriter, op.getLoc(), paramTensor, ValueRange{}).getResult();
  }

  template <typename QCOp, std::size_t... controlOperandIndices, std::size_t... targetOperandIndices,
            std::size_t... paramOperandIndices>
  LogicalResult convertGate(jasp::QuantumGateOp op, jasp::QuantumGateOp::Adaptor adaptor,
                            ConversionPatternRewriter& rewriter, std::index_sequence<controlOperandIndices...>,
                            std::index_sequence<targetOperandIndices...>,
                            std::index_sequence<paramOperandIndices...>) const {
    auto operands = adaptor.getGateOperands();
    const auto n_controls = sizeof...(controlOperandIndices);
    const auto n_targets = sizeof...(targetOperandIndices);
    const auto n_params = sizeof...(paramOperandIndices);

    const auto targetIndexOffset = n_controls;
    const auto paramIndexOffset = targetIndexOffset + n_targets;

    assert(operands.size() == n_controls + n_targets + n_params && "Invalid number of gate operands");

    if constexpr (n_controls == 0) {
      QCOp::create(rewriter, op.getLoc(), operands[targetOperandIndices]...,
                   extractParam(op, operands[paramOperandIndices + paramIndexOffset], rewriter)...);
    } else {
      qc::CtrlOp::create(rewriter, op.getLoc(), operands.take_front(n_controls), [&]() {
        QCOp::create(rewriter, op.getLoc(), operands[targetOperandIndices + targetIndexOffset]...,
                     extractParam(op, operands[paramOperandIndices + paramIndexOffset], rewriter)...);
      });
    }

    rewriter.eraseOp(op);
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
    auto tensorResult = tensor::FromElementsOp::create(rewriter, op.getLoc(), op.getType(0), measureBit);

    rewriter.replaceOpWithMultiple(op, {tensorResult.getResult(), ValueRange()});

    return success();
  }
};

/**
 * @brief Converts jasp.delete_qubits to memref.dealloc
 *
 * @details
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
    memref::DeallocOp::create(rewriter, op.getLoc(), array);
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Pass implementation for jasp-to-QC conversion
 *
 * @details
 * This pass converts the Jasp dialect to the QC dialect. It handles the
 * transformation of quantum state management from a functional-style
 * (QuantumState passing) to a side-effecting model. It also lowers qubit
 * array management to memref allocations and deallocations.
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
    target.addLegalDialect<QCDialect, memref::MemRefDialect, tensor::TensorDialect, arith::ArithDialect>();

    // Register operation conversion patterns
    patterns.add<ConvertJaspCreateQuantumKernelOp, ConvertJaspConsumeQuantumKernelOp, ConvertJaspCreateQubitsOp,
                 ConvertJaspGetQubitOp, ConvertJaspQuantumGateOp, ConvertJaspMeasureOp, ConvertJaspDeleteQubitsOp>(
        typeConverter, context);

    // Conversion of jasp types in func.func signatures
    // Note: This currently has limitations with signature changes
    // TODO: Find solution that works for 1-to-0 conversion of jasp:QuantumStateType
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
