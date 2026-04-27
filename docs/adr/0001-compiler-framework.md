---
status: "accepted"
date: 2026-04-14
decision-makers:
  - "Enrique Naranjo Bejarano"
  - "Julian Farnsteiner"
  - "Reinhard Stahn"
  - "Lukas Burgholzer"
  - "Yannick Stade"
---

# Compiler Framework

## Context and Problem Statement

As quantum computing systems become more complex and approach production use, there is a need for an industry-grade compiler framework capable of meeting the highly demanding requirements of quantum program execution and control. Our goal is to build a quantum compiler framework by leveraging an established classical compiler infrastructure.

The framework should be open source to enable collaboration from the broader quantum software community and facilitate integration by hardware manufacturers into their compilation pipelines. It must also support integration into high-performance computing (HPC) environments, which are expected to be early execution platforms for hybrid quantum-classical workflows.

The architecture must be extensible across both front end and back end layers and support the requirements of fault-tolerant quantum computing systems, including low-latency control flow execution.

On the front end, the framework must represent high-level quantum programming abstractions and progressively lower them through multiple levels of intermediate representations to machine-level code executed by quantum control hardware. It should support quantum programs with classical control flow, mid-circuit measurement, and dynamic circuit adaptation. It must also support kernel-based compilation workflows targeting heterogeneous classical accelerators (CPU, GPU, FPGA, etc.) used in hybrid execution stacks.

## Considered Options

- Develop a custom end-to-end compiler framework.
- Use LLVM as back end and MLIR as the primary intermediate representation / front end infrastructure.
- Use LLVM as back end with a custom-designed front end IR.

## Decision Outcome and Justification

We select LLVM as the back end compiler infrastructure and MLIR as the primary intermediate representation framework for the front end and mid-level compilation layers.

LLVM is a widely adopted, mature open-source compiler infrastructure that provides robust code generation capabilities across a broad range of target architectures, including those used in heterogeneous and HPC environments. It is actively maintained and broadly used in industry and research contexts. While LLVM provides strong back end capabilities, it imposes constraints in terms of IR flexibility and requires careful version management due to limited long-term ABI stability guarantees across releases. These constraints are manageable but require explicit dependency control and integration discipline.

Alternative compiler back ends such as GCC were considered but found to be less suitable for modular compiler construction due to their more monolithic architecture and comparatively limited support for custom IR pipelines and extensibility.

MLIR ("multi-level intermediate representation") is selected as the primary front end and mid-level IR infrastructure. It provides a flexible framework designed to support domain-specific IR language fragments ("dialects") and progressive lowering. This makes it particularly well suited for representing quantum-specific abstractions while integrating with LLVM-based back end pipelines.

MLIR enables the definition of custom dialects that can encode quantum operations, control flow, and hardware-specific constraints, and supports structured transformation and optimization passes across abstraction levels. This is particularly relevant for quantum compilation, where multiple representation layers are required between high-level quantum programs, mid-level circuit descriptions, and low-level control instructions.

A key design consideration is that certain quantum-specific semantics (e.g., linear resource constraints, uncomputation, and reversibility requirements) are not natively represented as first-class constructs in MLIR and must instead be enforced through dialect design and compiler transformations. While this introduces implementation complexity, it is considered acceptable given the maturity of the ecosystem and its extensibility.

We considered a fully custom frontend, a path taken by projects like [HUGR](https://github.com/Quantinuum/hugr). While HUGR provides a specialized graph-based IR for quantum computing, we declined this route for three reasons:

- Maintenance Overhead: A custom implementation like HUGR effectively recreates MLIR's core functionality (e.g., pass managers, verifiers) without the benefit of the global LLVM contributor base.
- Ecosystem Fragmentation: A custom IR limits direct interoperability with the vast library of existing classical LLVM/MLIR tooling, increasing the "integration tax" for hybrid applications.
- Resource Allocation: We prioritize developing quantum-specific optimizations over maintaining foundational IR infrastructure.

## Consequences

Positive:
- LLVM and MLIR provide a mature, actively maintained, and widely adopted compiler infrastructure suitable for HPC and heterogeneous execution environments.
- MLIR enables modular design with domain-specific dialects, which is well aligned with quantum compilation requirements.
- Leveraging LLVM/MLIR reduces the need to develop and maintain a full compiler backend stack.

Negative:
- LLVM versioning and upgrade coordination introduces integration overhead due to limited API and ABI stability.
- Extending LLVM backends or integrating tightly with low-level code generation pipelines can be complex and requires a skilled work-force.
- Quantum-specific semantics must be explicitly encoded and enforced through MLIR dialect design and transformation logic rather than being inherently supported by the underlying infrastructure.

## More Information

This decision is informed by prior experience from early development partners, including ParityQC, the Technical University of Munich (TUM), and Munich Quantum Software Company, who have successfully used LLVM/MLIR-based infrastructure in quantum compiler development efforts.
