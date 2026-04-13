#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "qcc/Conversion/JaspToQC/JaspToQC.h"
#include "qcc/Dialect/Jasp/IR/Jasp.h"

int main(int argc, char** argv) {

  // Dialect registration
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::bufferization::BufferizationDialect, mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  jasp::JaspDialect, mlir::qc::QCDialect>();

  // Pass registration
  qcc::registerJaspToQC();
  mlir::registerConvertLinalgToLoopsPass();
  mlir::bufferization::registerEmptyTensorToAllocTensorPass();
  mlir::bufferization::registerOneShotBufferizePass();
  mlir::registerLinalgDetensorizePass();
  mlir::registerCanonicalizer();
  mlir::registerCSEPass();
  mlir::bufferization::registerBufferLoopHoistingPass();
  mlir::registerMem2RegPass();
  mlir::registerSCCP();
  mlir::bufferization::registerPromoteBuffersToStackPass();
  mlir::registerInlinerPass();

  // Extension registration
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::func::registerInlinerExtension(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "qcc optimizer", registry));
}
