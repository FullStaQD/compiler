// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#ifndef INCLUDE_QCC_DIALECT_QCC_IR_QCCINTERFACES_TD_
#define INCLUDE_QCC_DIALECT_QCC_IR_QCCINTERFACES_TD_

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TopologyRewriteOpInterface
//===----------------------------------------------------------------------===//

def QCC_TopologyRewriteOpInterface : OpInterface<"TopologyRewriteOpInterface"> {
  let description = [{
    A Layer 3 rule: one double-pushout span `L <- I -> R` over the
    configuration, expressed as an operation that consumes a `!qcc.config` and
    produces its successor.

    Implementing this interface is what makes an operation part of a device's
    rule set. A device with an empty rule set, such as a superconducting chip,
    has no operation implementing it, so every `!qcc.config` use resolves to the
    value produced by `qcc.config.init` and the chain carries no dynamics. This
    is the case in which a static coupling graph is a sufficient model.

    The interface is intentionally small. The QCCD rule set is fixed and short,
    so each rule is a dedicated operation with its own verifier and cost model
    rather than an instance of a generic graph-rewrite engine.
  }];

  let cppNamespace = "::qcc::conn";

  let methods = [
    InterfaceMethod<
      "The configuration this rule rewrites — the `L` of the span.",
      "::mlir::Value", "getInputConfig", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getIn(); }]>,

    InterfaceMethod<
      "The rewritten configuration — the `R` of the span.",
      "::mlir::Value", "getOutputConfig", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getOut(); }]>,

    InterfaceMethod<[{
        Apply the rule to `in`, yielding the successor configuration.

        Returns failure and emits a diagnostic on `$_op` when the rule does not
        apply — no match for `L`, or a physical precondition such as a site
        capacity or a chain end is violated. Callers that only want to test
        applicability should suppress diagnostics around the call.
      }],
      "::mlir::FailureOr<::qcc::conn::Configuration>", "applyRule",
      (ins "const ::qcc::conn::Configuration &":$in)>,

    InterfaceMethod<[{
        The sites this rule touches and the control resources it contends for.

        Two rules whose sites are disjoint are parallel-independent: their
        matches meet only inside their interfaces, so applying them in either
        order yields the same configuration, and they may be reordered.

        Sharing a time step is a strictly stronger condition, and the footprint
        carries the contended control resources for it. A control resource is
        control-plane occupancy rather than a property of the configuration, so
        it constrains concurrency without constraining order; `commutesWith` and
        `concurrentWith` keep the two questions apart.
      }],
      "::qcc::conn::Footprint", "getFootprint", (ins)>,

    InterfaceMethod<
      "Cost of the rule in device-native units (microseconds).",
      "double", "getLatencyUs", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return 0.0; }]>,
  ];

  let extraSharedClassDeclaration = [{
    /// Whether this rule and `other` may be reordered with respect to one
    /// another. This is the double-pushout question and reads only the sites
    /// the two footprints touch; a shared control resource does not prevent
    /// reordering, only concurrency.
    bool commutesWith(TopologyRewriteOpInterface other) {
      return $_op.getFootprint().commutesWith(other.getFootprint());
    }
    /// Whether this rule and `other` may occupy the same time step, which
    /// additionally requires that they contend for no control resource.
    bool concurrentWith(TopologyRewriteOpInterface other) {
      return $_op.getFootprint().concurrentWith(other.getFootprint());
    }
    /// The device the rewritten configuration is typed over.
    ::qcc::conn::DeviceAttr getDeviceAttr() {
      return ::llvm::cast<::qcc::conn::ConfigType>($_op.getOutputConfig().getType()).getDevice();
    }
  }];
}

#endif // INCLUDE_QCC_DIALECT_QCC_IR_QCCINTERFACES_TD_
