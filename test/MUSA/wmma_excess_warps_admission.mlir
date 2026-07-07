// RUN: triton-opt %s --tritonmusa-accelerate-matmul | FileCheck %s

// CHECK: #[[$MMA:.+]] = #ttg.musa_wmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [8, 1], instrShape = [16, 16, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @wmma_excess_warps_admission
  // CHECK: ttmg.wmma_dot {{.*}} {eltTypeA = 6 : i32, eltTypeB = 6 : i32, k = 16 : i32, {{.*}}m = 16 : i32, n = 16 : i32
  // CHECK-SAME: tensor<32x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[$MMA]]}>> * tensor<64x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[$MMA]]}>> -> tensor<32x16xf32, #[[$MMA]]>
  tt.func public @wmma_excess_warps_admission(
      %a: tensor<32x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %b: tensor<64x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %acc: tensor<32x16xf32, #blocked>) -> tensor<32x16xf32, #blocked> {
    %dot = tt.dot %a, %b, %acc, inputPrecision = tf32 :
      tensor<32x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> *
      tensor<64x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> ->
      tensor<32x16xf32, #blocked>
    tt.return %dot : tensor<32x16xf32, #blocked>
  }
}
