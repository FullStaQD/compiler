#include "mlir/IR/MLIRContext.h"
#include <iostream>

int main() {
    mlir::MLIRContext context;
    std::cout << "qcc: MLIR context initialized successfully." << std::endl;
    return 0;
}