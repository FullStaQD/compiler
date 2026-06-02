// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def AuxOutputRecording : Pass<"integer-output-recording"> {
    let summary = "Add recording of integer outputs.";
    let description = [{
        This pass adds `aux.record_integer` operations after every operation that returns an integer value.
    }];
    let summary = "Adds recording of integer outputs following the QIR recipe for output recording.";
    let description = [{
      This pass adds `aux.record_integer` operations after every operation that returns an integer value, following the QIR recipe for output recording. This allows us to record integer outputs in a way that can be translated to the corresponding QIR runtime function calls during code generation.

      For example, if we have an operation that produces an integer result:

      ```mlir
      %result = some_operation() : () -> i64
      return %result : i64
      ```

      After applying this pass, it will be transformed to:

      ```mlir
      %result = some_operation() : () -> i64
      aux.record_integer %result : i64
      return
      ```
    This pass is a workaround to enable output recording for integer values, as our principal
    dialects do not have a direct way to represent the necessary functionality for recording outputs.
   }];
    let dependentDialects = [
    "mlir::func::FuncDialect",
    "::qcc::aux::AuxDialect"
    ];
}
