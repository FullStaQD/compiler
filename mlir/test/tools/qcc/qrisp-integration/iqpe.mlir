// RUN: qcc %s -o %t.ll
// RUN: FileCheck %s --check-prefix=CHECK-QIR < %t.ll
// RUN: qir-runner --file %t.ll -s 5 | FileCheck %s --check-prefix=CHECK-SIM

// GENERATED FROM QRISP VERSION 0.9.5

builtin.module @jasp_module {
  func.func public @main(%arg18: !jasp.QuantumState) -> (tensor<i64>, !jasp.QuantumState) {
    %0 = arith.constant dense<1> : tensor<i64>
    %1, %2 = jasp.create_qubits %0, %arg18 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    %3 = arith.constant dense<0> : tensor<i64>
    %4 = jasp.get_qubit %1, %3 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %5 = jasp.quantum_gate "x" (%4) , %2 : (!jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
    %6, %7 = jasp.create_qubits %0, %5 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    %8 = arith.constant dense<3.125000e-01> : tensor<f64>
    %9 = arith.constant dense<4> : tensor<i64>
    %10, %11, %12, %13, %14, %15, %16, %17 = scf.while (%arg229 = %6, %arg230 = %8, %arg231 = %1, %arg232 = %9, %arg233 = %3, %arg234 = %3, %arg235 = %9, %arg236 = %7) : (!jasp.QubitArray, tensor<f64>, !jasp.QubitArray, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, tensor<f64>, !jasp.QubitArray, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, !jasp.QuantumState) {
      %18 = tensor.extract %arg234[] : tensor<i64>
      %19 = tensor.extract %arg235[] : tensor<i64>
      %20 = arith.cmpi slt, %18, %19 : i64
      scf.condition(%20) %arg229, %arg230, %arg231, %arg232, %arg233, %arg234, %arg235, %arg236 : !jasp.QubitArray, tensor<f64>, !jasp.QubitArray, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, !jasp.QuantumState
    } do {
    ^bb0(%arg19: !jasp.QubitArray, %arg20: tensor<f64>, %arg21: !jasp.QubitArray, %arg22: tensor<i64>, %arg23: tensor<i64>, %arg24: tensor<i64>, %arg25: tensor<i64>, %arg26: !jasp.QuantumState):
      %21 = arith.constant dense<0> : tensor<i64>
      %22 = jasp.get_qubit %arg19, %21 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
      %23 = jasp.reset %22, %arg26 : !jasp.Qubit, !jasp.QuantumState -> !jasp.QuantumState
      %24 = jasp.quantum_gate "h" (%22) , %23 : (!jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
      %25 = arith.constant 6.2831853071795862 : f64
      %26 = tensor.extract %arg20[] : tensor<f64>
      %27 = arith.mulf %25, %26 : f64
      %28 = tensor.extract %arg22[] : tensor<i64>
      %29 = tensor.extract %arg24[] : tensor<i64>
      %30 = arith.subi %28, %29 : i64
      %31 = arith.constant dense<1> : tensor<i64>
      %32 = arith.constant 1 : i64
      %33 = arith.subi %30, %32 : i64
      %34 = arith.constant 2 : i64
      %35 = arith.constant 0 : i64
      %36 = arith.cmpi eq, %34, %35 : i64
      %37 = arith.constant 0 : i64
      %38 = arith.cmpi ne, %33, %37 : i64
      %39 = arith.andi %36, %38 : i1
      %40 = tensor.from_elements %39 : tensor<i1>
      %41 = func.call @_where(%40, %21, %31) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
      %42 = arith.constant 1 : i64
      %43 = arith.andi %33, %42 : i64
      %44 = tensor.from_elements %43 : tensor<i64>
      %45 = arith.constant 2 : i64
      %46 = tensor.extract %41[] : tensor<i64>
      %47 = arith.muli %46, %45 : i64
      %48 = tensor.from_elements %47 : tensor<i64>
      %49 = func.call @_where_4(%44, %48, %41) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
      %50 = arith.constant 4 : i64
      %51 = arith.constant 1 : i64
      %52 = arith.constant 0 : i64
      %53 = arith.shrui %33, %51 : i64
      %54 = arith.constant 64 : i64
      %55 = arith.cmpi ugt, %54, %51 : i64
      %56 = arith.select %55, %53, %52 : i64
      %57 = arith.constant 1 : i64
      %58 = arith.andi %56, %57 : i64
      %59 = tensor.from_elements %58 : tensor<i64>
      %60 = tensor.extract %49[] : tensor<i64>
      %61 = arith.muli %60, %50 : i64
      %62 = tensor.from_elements %61 : tensor<i64>
      %63 = func.call @_where_4(%59, %62, %49) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
      %64 = arith.constant 16 : i64
      %65 = arith.constant 1 : i64
      %66 = arith.constant 0 : i64
      %67 = arith.shrui %56, %65 : i64
      %68 = arith.constant 64 : i64
      %69 = arith.cmpi ugt, %68, %65 : i64
      %70 = arith.select %69, %67, %66 : i64
      %71 = arith.constant 1 : i64
      %72 = arith.andi %70, %71 : i64
      %73 = tensor.from_elements %72 : tensor<i64>
      %74 = tensor.extract %63[] : tensor<i64>
      %75 = arith.muli %74, %64 : i64
      %76 = tensor.from_elements %75 : tensor<i64>
      %77 = func.call @_where_4(%73, %76, %63) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
      %78 = arith.constant 256 : i64
      %79 = arith.constant 1 : i64
      %80 = arith.constant 0 : i64
      %81 = arith.shrui %70, %79 : i64
      %82 = arith.constant 64 : i64
      %83 = arith.cmpi ugt, %82, %79 : i64
      %84 = arith.select %83, %81, %80 : i64
      %85 = arith.constant 1 : i64
      %86 = arith.andi %84, %85 : i64
      %87 = tensor.from_elements %86 : tensor<i64>
      %88 = tensor.extract %77[] : tensor<i64>
      %89 = arith.muli %88, %78 : i64
      %90 = tensor.from_elements %89 : tensor<i64>
      %91 = func.call @_where_4(%87, %90, %77) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
      %92 = arith.constant 65536 : i64
      %93 = arith.constant 1 : i64
      %94 = arith.constant 0 : i64
      %95 = arith.shrui %84, %93 : i64
      %96 = arith.constant 64 : i64
      %97 = arith.cmpi ugt, %96, %93 : i64
      %98 = arith.select %97, %95, %94 : i64
      %99 = arith.constant 1 : i64
      %100 = arith.andi %98, %99 : i64
      %101 = tensor.from_elements %100 : tensor<i64>
      %102 = tensor.extract %91[] : tensor<i64>
      %103 = arith.muli %102, %92 : i64
      %104 = tensor.from_elements %103 : tensor<i64>
      %105 = func.call @_where_4(%101, %104, %91) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
      %106 = arith.constant 4294967296 : i64
      %107 = arith.constant 1 : i64
      %108 = arith.constant 0 : i64
      %109 = arith.shrui %98, %107 : i64
      %110 = arith.constant 64 : i64
      %111 = arith.cmpi ugt, %110, %107 : i64
      %112 = arith.select %111, %109, %108 : i64
      %113 = arith.constant 1 : i64
      %114 = arith.andi %112, %113 : i64
      %115 = tensor.from_elements %114 : tensor<i64>
      %116 = tensor.extract %105[] : tensor<i64>
      %117 = arith.muli %116, %106 : i64
      %118 = tensor.from_elements %117 : tensor<i64>
      %119 = func.call @_where_4(%115, %118, %105) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
      %120 = tensor.extract %119[] : tensor<i64>
      %121 = arith.sitofp %120 : i64 to f64
      %122 = arith.mulf %27, %121 : f64
      %123 = jasp.get_qubit %arg21, %21 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
      %124 = arith.constant 5.000000e-01 : f64
      %125 = arith.mulf %124, %122 : f64
      %126 = tensor.from_elements %125 : tensor<f64>
      %127 = jasp.quantum_gate "p" (%123, %126) , %24 : (!jasp.Qubit, tensor<f64>) , !jasp.QuantumState -> !jasp.QuantumState
      %128 = arith.constant 5.000000e-01 : f64
      %129 = arith.mulf %128, %122 : f64
      %130 = tensor.from_elements %129 : tensor<f64>
      %131 = jasp.quantum_gate "p" (%22, %130) , %127 : (!jasp.Qubit, tensor<f64>) , !jasp.QuantumState -> !jasp.QuantumState
      %132 = jasp.quantum_gate "cx" (%22, %123) , %131 : (!jasp.Qubit, !jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
      %133 = arith.constant -5.000000e-01 : f64
      %134 = arith.mulf %133, %122 : f64
      %135 = tensor.from_elements %134 : tensor<f64>
      %136 = jasp.quantum_gate "p" (%123, %135) , %132 : (!jasp.Qubit, tensor<f64>) , !jasp.QuantumState -> !jasp.QuantumState
      %137 = jasp.quantum_gate "cx" (%22, %123) , %136 : (!jasp.Qubit, !jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
      %138 = tensor.extract %arg24[] : tensor<i64>
      %139 = arith.constant 1 : i64
      %140 = arith.subi %138, %139 : i64
      %141 = tensor.from_elements %140 : tensor<i64>
      %142 = arith.subi %140, %140 : i64
      %143 = tensor.from_elements %142 : tensor<i64>
      %144, %145, %146, %147, %148, %149 = scf.while (%arg132 = %arg23, %arg133 = %arg24, %arg134 = %arg19, %arg135 = %141, %arg136 = %143, %arg137 = %137) : (tensor<i64>, tensor<i64>, !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) -> (tensor<i64>, tensor<i64>, !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) {
        %150 = tensor.extract %arg136[] : tensor<i64>
        %151 = tensor.extract %arg135[] : tensor<i64>
        %152 = arith.cmpi sle, %150, %151 : i64
        scf.condition(%152) %arg132, %arg133, %arg134, %arg135, %arg136, %arg137 : tensor<i64>, tensor<i64>, !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
      } do {
      ^bb1(%arg39: tensor<i64>, %arg40: tensor<i64>, %arg41: !jasp.QubitArray, %arg42: tensor<i64>, %arg43: tensor<i64>, %arg44: !jasp.QuantumState):
        %153 = tensor.extract %arg39[] : tensor<i64>
        %154 = tensor.extract %arg43[] : tensor<i64>
        %155 = arith.constant 63 : i64
        %156 = arith.shrsi %153, %155 : i64
        %157 = arith.shrsi %153, %154 : i64
        %158 = arith.constant 64 : i64
        %159 = arith.cmpi ugt, %158, %154 : i64
        %160 = arith.select %159, %157, %156 : i64
        %161 = arith.constant 1 : i64
        %162 = arith.andi %160, %161 : i64
        %163 = arith.constant 1 : i64
        %164 = arith.cmpi eq, %162, %163 : i64
        %165 = arith.constant true
        %166 = arith.xori %164, %165 : i1
        %167 = scf.if %166 -> (!jasp.QuantumState) {
          scf.yield %arg44 : !jasp.QuantumState
        } else {
          %168 = tensor.extract %arg40[] : tensor<i64>
          %169 = tensor.extract %arg43[] : tensor<i64>
          %170 = arith.subi %169, %168 : i64
          %171 = arith.constant dense<1> : tensor<i64>
          %172 = arith.constant 1 : i64
          %173 = arith.subi %170, %172 : i64
          %174 = arith.constant dense<0> : tensor<i64>
          %175 = arith.constant 2 : i64
          %176 = arith.constant 0 : i64
          %177 = arith.cmpi eq, %175, %176 : i64
          %178 = arith.constant 0 : i64
          %179 = arith.cmpi ne, %173, %178 : i64
          %180 = arith.andi %177, %179 : i1
          %181 = tensor.from_elements %180 : tensor<i1>
          %182 = func.call @_where(%181, %174, %171) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
          %183 = arith.constant 1 : i64
          %184 = arith.andi %173, %183 : i64
          %185 = tensor.from_elements %184 : tensor<i64>
          %186 = arith.constant 2 : i64
          %187 = tensor.extract %182[] : tensor<i64>
          %188 = arith.muli %187, %186 : i64
          %189 = tensor.from_elements %188 : tensor<i64>
          %190 = func.call @_where_4(%185, %189, %182) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
          %191 = arith.constant 4 : i64
          %192 = arith.constant 1 : i64
          %193 = arith.constant 0 : i64
          %194 = arith.shrui %173, %192 : i64
          %195 = arith.constant 64 : i64
          %196 = arith.cmpi ugt, %195, %192 : i64
          %197 = arith.select %196, %194, %193 : i64
          %198 = arith.constant 1 : i64
          %199 = arith.andi %197, %198 : i64
          %200 = tensor.from_elements %199 : tensor<i64>
          %201 = tensor.extract %190[] : tensor<i64>
          %202 = arith.muli %201, %191 : i64
          %203 = tensor.from_elements %202 : tensor<i64>
          %204 = func.call @_where_4(%200, %203, %190) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
          %205 = arith.constant 16 : i64
          %206 = arith.constant 1 : i64
          %207 = arith.constant 0 : i64
          %208 = arith.shrui %197, %206 : i64
          %209 = arith.constant 64 : i64
          %210 = arith.cmpi ugt, %209, %206 : i64
          %211 = arith.select %210, %208, %207 : i64
          %212 = arith.constant 1 : i64
          %213 = arith.andi %211, %212 : i64
          %214 = tensor.from_elements %213 : tensor<i64>
          %215 = tensor.extract %204[] : tensor<i64>
          %216 = arith.muli %215, %205 : i64
          %217 = tensor.from_elements %216 : tensor<i64>
          %218 = func.call @_where_4(%214, %217, %204) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
          %219 = arith.constant 256 : i64
          %220 = arith.constant 1 : i64
          %221 = arith.constant 0 : i64
          %222 = arith.shrui %211, %220 : i64
          %223 = arith.constant 64 : i64
          %224 = arith.cmpi ugt, %223, %220 : i64
          %225 = arith.select %224, %222, %221 : i64
          %226 = arith.constant 1 : i64
          %227 = arith.andi %225, %226 : i64
          %228 = tensor.from_elements %227 : tensor<i64>
          %229 = tensor.extract %218[] : tensor<i64>
          %230 = arith.muli %229, %219 : i64
          %231 = tensor.from_elements %230 : tensor<i64>
          %232 = func.call @_where_4(%228, %231, %218) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
          %233 = arith.constant 65536 : i64
          %234 = arith.constant 1 : i64
          %235 = arith.constant 0 : i64
          %236 = arith.shrui %225, %234 : i64
          %237 = arith.constant 64 : i64
          %238 = arith.cmpi ugt, %237, %234 : i64
          %239 = arith.select %238, %236, %235 : i64
          %240 = arith.constant 1 : i64
          %241 = arith.andi %239, %240 : i64
          %242 = tensor.from_elements %241 : tensor<i64>
          %243 = tensor.extract %232[] : tensor<i64>
          %244 = arith.muli %243, %233 : i64
          %245 = tensor.from_elements %244 : tensor<i64>
          %246 = func.call @_where_4(%242, %245, %232) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
          %247 = arith.constant 4294967296 : i64
          %248 = arith.constant 1 : i64
          %249 = arith.constant 0 : i64
          %250 = arith.shrui %239, %248 : i64
          %251 = arith.constant 64 : i64
          %252 = arith.cmpi ugt, %251, %248 : i64
          %253 = arith.select %252, %250, %249 : i64
          %254 = arith.constant 1 : i64
          %255 = arith.andi %253, %254 : i64
          %256 = tensor.from_elements %255 : tensor<i64>
          %257 = tensor.extract %246[] : tensor<i64>
          %258 = arith.muli %257, %247 : i64
          %259 = tensor.from_elements %258 : tensor<i64>
          %260 = func.call @_where_4(%256, %259, %246) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
          %261 = tensor.extract %260[] : tensor<i64>
          %262 = arith.sitofp %261 : i64 to f64
          %263 = arith.constant -6.2831853071795862 : f64
          %264 = arith.mulf %263, %262 : f64
          %265 = tensor.from_elements %264 : tensor<f64>
          %266 = jasp.get_qubit %arg41, %174 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
          %267 = jasp.quantum_gate "rz" (%266, %265) , %arg44 : (!jasp.Qubit, tensor<f64>) , !jasp.QuantumState -> !jasp.QuantumState
          scf.yield %267 : !jasp.QuantumState
        }
        %268 = arith.constant 1 : i64
        %269 = tensor.extract %arg43[] : tensor<i64>
        %270 = arith.addi %269, %268 : i64
        %271 = tensor.from_elements %270 : tensor<i64>
        %272 = func.call @_jrange_marker(%271, %arg42) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        scf.yield %arg39, %arg40, %arg41, %arg42, %272, %167 : tensor<i64>, tensor<i64>, !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
      }
      %273 = jasp.quantum_gate "h" (%22) , %149 : (!jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
      %274, %275 = jasp.measure %arg19, %273 : !jasp.QubitArray, !jasp.QuantumState -> tensor<i64>, !jasp.QuantumState
      %276 = tensor.extract %arg24[] : tensor<i64>
      %277 = tensor.extract %274[] : tensor<i64>
      %278 = arith.constant 0 : i64
      %279 = arith.shli %277, %276 : i64
      %280 = arith.constant 64 : i64
      %281 = arith.cmpi ugt, %280, %276 : i64
      %282 = arith.select %281, %279, %278 : i64
      %283 = tensor.extract %arg23[] : tensor<i64>
      %284 = arith.ori %283, %282 : i64
      %285 = tensor.from_elements %284 : tensor<i64>
      %286 = arith.constant 1 : i64
      %287 = tensor.extract %arg24[] : tensor<i64>
      %288 = arith.addi %287, %286 : i64
      %289 = tensor.from_elements %288 : tensor<i64>
      scf.yield %arg19, %arg20, %arg21, %arg22, %285, %289, %arg25, %275 : !jasp.QubitArray, tensor<f64>, !jasp.QubitArray, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, !jasp.QuantumState
    }
    func.return %14, %17 : tensor<i64>, !jasp.QuantumState
  }
  func.func private @_where(%arg11: tensor<i1>, %arg12: tensor<i64>, %arg13: tensor<i64>) -> (tensor<i64>) {
    %0 = tensor.extract %arg11[] : tensor<i1>
    %1 = tensor.extract %arg12[] : tensor<i64>
    %2 = tensor.extract %arg13[] : tensor<i64>
    %3 = arith.select %0, %1, %2 : i64
    %4 = tensor.from_elements %3 : tensor<i64>
    func.return %4 : tensor<i64>
  }
  func.func private @_where_4(%arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<i64>) -> (tensor<i64>) {
    %0 = arith.constant 0 : i64
    %1 = tensor.extract %arg2[] : tensor<i64>
    %2 = arith.cmpi ne, %1, %0 : i64
    %3 = tensor.extract %arg3[] : tensor<i64>
    %4 = tensor.extract %arg4[] : tensor<i64>
    %5 = arith.select %2, %3, %4 : i64
    %6 = tensor.from_elements %5 : tensor<i64>
    func.return %6 : tensor<i64>
  }
  func.func private @_jrange_marker(%arg0: tensor<i64>, %arg1: tensor<i64>) -> (tensor<i64>) {
    func.return %arg0 : tensor<i64>
  }
}


// At precision 4, we expect four measurements of the same auxiliary qubit 1.
// CHECK-QIR: call void @__quantum__qis__mz__body(ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 1 to ptr))
// CHECK-QIR: call void @__quantum__qis__mz__body(ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 1 to ptr))
// CHECK-QIR: call void @__quantum__qis__mz__body(ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 1 to ptr))
// CHECK-QIR: call void @__quantum__qis__mz__body(ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 1 to ptr))
// CHECK-QIR-NOT: call void @__quantum__qis__mz__body


// CHECK-SIM: START
// CHECK-SIM: METADATA required_num_qubits 2
// CHECK-SIM: METADATA required_num_results 2
// Expected output phase is 0.3125, in binary 0.0101, encoded as integer 0b 0101 = 5.
// CHECK-SIM: OUTPUT INT 5 dummy_label
// CHECK-SIM: OUTPUT INT 5 dummy_label
// CHECK-SIM: OUTPUT INT 5 dummy_label
// CHECK-SIM: OUTPUT INT 5 dummy_label
// CHECK-SIM: OUTPUT INT 5 dummy_label
// CHECK-SIM: END 0
