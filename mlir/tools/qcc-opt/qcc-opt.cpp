#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "qcc/Dialect/Dummy/IR/Dummy.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/QC/IR/QCDialect.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<
        mlir::func::FuncDialect,
        mlir::arith::ArithDialect,
        qcc::dummy::DummyDialect,
        mlir::qc::QCDialect
    >();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "qcc optimizer", registry)
    );
}