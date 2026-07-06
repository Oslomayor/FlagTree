// RUN: triton-opt %s --tritonmusa-accelerate-matmul | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
#mma_anchor = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 16]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @register_musa_dot_dialect(%a_smem: !ttg.memdesc<32x16xf32, #shared_a, #smem, mutable>, %b_smem: !ttg.memdesc<16x32xf32, #shared_b, #smem, mutable>, %acc: tensor<32x32xf32, #mma_anchor>) -> tensor<32x32xf32, #mma_anchor> {
    %false = arith.constant false
    %dot = ttmg.squad_dot %a_smem, %b_smem, %acc, %false {eltTypeA = 6 : i32, eltTypeB = 6 : i32, eltTypeC = 7 : i32, inputPrecision = 0 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf32, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf32, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma_anchor>
    tt.return %dot : tensor<32x32xf32, #mma_anchor>
  }

  // CHECK-LABEL: tt.func public @sqmma_f32_plain_transpose_b
  // CHECK: %[[B_T:.+]] = tt.trans {{.*}} : tensor<32x64xf32, {{#[A-Za-z0-9_]+}}> -> tensor<64x32xf32, {{#[A-Za-z0-9_]+}}>
  // CHECK: %[[B_SMEM:.+]] = ttg.local_alloc %[[B_T]] {{.*}} : (tensor<64x32xf32, {{#[A-Za-z0-9_]+}}>) -> !ttg.memdesc<64x32xf32, {{#[A-Za-z0-9_]+}}, #smem, mutable>
  // CHECK: ttmg.squad_dot {{.*}}, %[[B_SMEM]], {{.*}} : !ttg.memdesc<32x64xf32, {{#[A-Za-z0-9_]+}}, #smem, mutable> * !ttg.memdesc<64x32xf32, {{#[A-Za-z0-9_]+}}, #smem, mutable> -> tensor<32x32xf32, {{#[A-Za-z0-9_]+}}>
  tt.func public @sqmma_f32_plain_transpose_b(%a: tensor<32x64xf32, #dot_a>, %b: tensor<32x64xf32, #blocked>, %acc: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %b_t_blocked = tt.trans %b {order = array<i32: 1, 0>} : tensor<32x64xf32, #blocked> -> tensor<64x32xf32, #blocked_t>
    %b_t = ttg.convert_layout %b_t_blocked : tensor<64x32xf32, #blocked_t> -> tensor<64x32xf32, #dot_b>
    %dot = tt.dot %a, %b_t, %acc, inputPrecision = tf32 : tensor<32x64xf32, #dot_a> * tensor<64x32xf32, #dot_b> -> tensor<32x32xf32, #blocked>
    tt.return %dot : tensor<32x32xf32, #blocked>
  }
}
