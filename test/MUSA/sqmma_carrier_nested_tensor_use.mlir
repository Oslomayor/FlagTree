// RUN: triton-opt %s --tritonmusa-convert-sqmma-to-mtgpu | FileCheck %s

#mma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 16]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @sqmma_carrier_nested_tensor_use
  // CHECK: mtgpu.sqmma_wait
  // CHECK: scf.if
  // CHECK: %[[UNPACK:.+]] = mtgpu.unpack_sqmma_accumulator
  // CHECK-NEXT: arith.truncf %[[UNPACK]] : tensor<32x32xf32, #{{.*}}> to tensor<32x32xf16, #{{.*}}>
  // CHECK-NOT: arith.truncf {{.*}}!mtgpu.sqmma_accumulator
  tt.func public @sqmma_carrier_nested_tensor_use(
      %a_smem: !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>,
      %b_smem: !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>,
      %do_epilogue: i1) -> tensor<32x32xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %false = arith.constant false
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %result = scf.for %iv = %c0 to %c2 step %c1 iter_args(%acc = %zero) -> tensor<32x32xf32, #mma> {
      %dot = ttmg.squad_dot %a_smem, %b_smem, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, isAsync = true, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
      %wait:3 = ttmg.squad_dot_wait %dot, %a_smem, %b_smem : tensor<32x32xf32, #mma>, !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
      scf.if %do_epilogue {
        %narrow = arith.truncf %wait#0 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
        scf.yield
      }
      scf.yield %wait#0 : tensor<32x32xf32, #mma>
    }
    tt.return %result : tensor<32x32xf32, #mma>
  }
}
