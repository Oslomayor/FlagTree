// RUN: triton-opt %s --tritonmusa-accelerate-matmul | FileCheck %s --check-prefix=ACCEL
// RUN: triton-opt %s --tritonmusa-accelerate-matmul --tritonmusa-pipeline=num-stages=2 -canonicalize | FileCheck %s --check-prefix=PIPE --implicit-check-not=tt.load
// RUN: sed 's/f16/bf16/g' %s | triton-opt --tritonmusa-accelerate-matmul | FileCheck %s --check-prefix=BF16-ACCEL
// RUN: sed 's/f16/bf16/g' %s | triton-opt --tritonmusa-accelerate-matmul --tritonmusa-pipeline=num-stages=2 -canonicalize | FileCheck %s --check-prefix=BF16-PIPE --implicit-check-not=tt.load
// RUN: sed 's/musa:31/musa:ph1s/' %s | triton-opt --tritonmusa-accelerate-matmul | FileCheck %s --check-prefix=ACCEL
// RUN: sed -e 's/musa:31/musa:ph1s/' -e 's/f16/bf16/g' %s | triton-opt --tritonmusa-accelerate-matmul | FileCheck %s --check-prefix=BF16-ACCEL

#out_pv = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#out_qk = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#row = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#col = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#q_a = #ttg.dot_op<{opIdx = 0, parent = #out_qk}>
#k_b = #ttg.dot_op<{opIdx = 1, parent = #out_qk}>
#p_a = #ttg.dot_op<{opIdx = 0, parent = #out_pv}>
#v_b = #ttg.dot_op<{opIdx = 1, parent = #out_pv}>
#out_3d = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
#a_3d_src = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 8, 1], order = [1, 2, 0]}>
#b_3d_src = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 8, 1], order = [1, 2, 0]}>
#a_3d = #ttg.dot_op<{opIdx = 0, parent = #out_3d}>
#b_3d = #ttg.dot_op<{opIdx = 1, parent = #out_3d}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // ACCEL-LABEL: tt.func public @sqmma_mixed_source_orders
  // ACCEL: %[[K_LOAD:.+]] = tt.load {{.*}} : tensor<128x32x!tt.ptr<f16>, {{#[A-Za-z0-9_]+}}>
  // ACCEL: %[[K_SMEM:.+]] = ttg.local_alloc %[[K_LOAD]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false}
  // ACCEL: ttmg.squad_dot {{.*}}, %[[K_SMEM]], {{.*}} {
  // ACCEL-SAME: layoutB = 1 : i32
  // ACCEL: %[[V_LOAD:.+]] = tt.load {{.*}} : tensor<32x128x!tt.ptr<f16>, {{#[A-Za-z0-9_]+}}>
  // ACCEL: %[[V_SMEM:.+]] = ttg.local_alloc %[[V_LOAD]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true}
  // ACCEL: ttmg.squad_dot {{.*}}, %[[V_SMEM]], {{.*}} {
  // ACCEL-SAME: layoutB = 0 : i32

  // PIPE-LABEL: tt.func public @sqmma_mixed_source_orders
  // PIPE-DAG: %[[K_ROOT:.+]] = ttg.local_alloc {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false} : () -> !ttg.memdesc<2x128x32xf16
  // PIPE-DAG: %[[V_ROOT:.+]] = ttg.local_alloc {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : () -> !ttg.memdesc<2x32x128xf16
  // PIPE: %[[K_DST:.+]] = ttg.memdesc_index %[[K_ROOT]]
  // PIPE: ttg.async_copy_global_to_local {{.*}}, %[[K_DST]] {{.*}}{contiguity = 8 : i32}
  // PIPE-SAME: : tensor<128x32x!tt.ptr<f16>,
  // PIPE-SAME: -> <128x32xf16,
  // PIPE: %[[V_DST:.+]] = ttg.memdesc_index %[[V_ROOT]]
  // PIPE: ttg.async_copy_global_to_local {{.*}}, %[[V_DST]] {{.*}}{contiguity = 8 : i32}
  // PIPE-SAME: : tensor<32x128x!tt.ptr<f16>,
  // PIPE-SAME: -> <32x128xf16,

  // BF16-ACCEL-LABEL: tt.func public @sqmma_mixed_source_orders
  // BF16-ACCEL: %[[BF16_K_LOAD:.+]] = tt.load {{.*}} : tensor<128x32x!tt.ptr<bf16>, {{#[A-Za-z0-9_]+}}>
  // BF16-ACCEL: %[[BF16_K_SMEM:.+]] = ttg.local_alloc %[[BF16_K_LOAD]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false}
  // BF16-ACCEL: ttmg.squad_dot {{.*}}, %[[BF16_K_SMEM]], {{.*}} {
  // BF16-ACCEL-SAME: layoutB = 1 : i32
  // BF16-ACCEL: %[[BF16_V_LOAD:.+]] = tt.load {{.*}} : tensor<32x128x!tt.ptr<bf16>, {{#[A-Za-z0-9_]+}}>
  // BF16-ACCEL: %[[BF16_V_SMEM:.+]] = ttg.local_alloc %[[BF16_V_LOAD]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true}
  // BF16-ACCEL: ttmg.squad_dot {{.*}}, %[[BF16_V_SMEM]], {{.*}} {
  // BF16-ACCEL-SAME: layoutB = 0 : i32

  // BF16-PIPE-LABEL: tt.func public @sqmma_mixed_source_orders
  // BF16-PIPE-DAG: %[[BF16_K_ROOT:.+]] = ttg.local_alloc {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false} : () -> !ttg.memdesc<2x128x32xbf16
  // BF16-PIPE-DAG: %[[BF16_V_ROOT:.+]] = ttg.local_alloc {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : () -> !ttg.memdesc<2x32x128xbf16
  // BF16-PIPE: %[[BF16_K_DST:.+]] = ttg.memdesc_index %[[BF16_K_ROOT]]
  // BF16-PIPE: ttg.async_copy_global_to_local {{.*}}, %[[BF16_K_DST]] {{.*}}{contiguity = 8 : i32}
  // BF16-PIPE-SAME: : tensor<128x32x!tt.ptr<bf16>,
  // BF16-PIPE-SAME: -> <128x32xbf16,
  // BF16-PIPE: %[[BF16_V_DST:.+]] = ttg.memdesc_index %[[BF16_V_ROOT]]
  // BF16-PIPE: ttg.async_copy_global_to_local {{.*}}, %[[BF16_V_DST]] {{.*}}{contiguity = 8 : i32}
  // BF16-PIPE-SAME: : tensor<32x128x!tt.ptr<bf16>,
  // BF16-PIPE-SAME: -> <32x128xbf16,
  tt.func public @sqmma_mixed_source_orders(
      %q: tensor<64x128xf16, #row>,
      %p: tensor<64x32xf16, #out_qk>,
      %kp: tensor<128x32x!tt.ptr<f16>, #col> {
        tt.divisibility = dense<[16, 16]> : tensor<2xi32>,
        tt.contiguity = dense<[16, 1]> : tensor<2xi32>},
      %vp: tensor<32x128x!tt.ptr<f16>, #row> {
        tt.divisibility = dense<[16, 16]> : tensor<2xi32>,
        tt.contiguity = dense<[1, 16]> : tensor<2xi32>},
      %n: index) -> (tensor<64x32xf32, #out_qk>, tensor<64x128xf32, #out_pv>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zqk = arith.constant dense<0.0> : tensor<64x32xf32, #out_qk>
    %zpv = arith.constant dense<0.0> : tensor<64x128xf32, #out_pv>
    %r:2 = scf.for %i = %c0 to %n step %c1
        iter_args(%aqk = %zqk, %apv = %zpv)
        -> (tensor<64x32xf32, #out_qk>, tensor<64x128xf32, #out_pv>) {
      %k = tt.load %kp : tensor<128x32x!tt.ptr<f16>, #col>
      %qd = ttg.convert_layout %q : tensor<64x128xf16, #row> -> tensor<64x128xf16, #q_a>
      %kd = ttg.convert_layout %k : tensor<128x32xf16, #col> -> tensor<128x32xf16, #k_b>
      %qk = tt.dot %qd, %kd, %aqk : tensor<64x128xf16, #q_a> * tensor<128x32xf16, #k_b> -> tensor<64x32xf32, #out_qk>

      %v = tt.load %vp : tensor<32x128x!tt.ptr<f16>, #row>
      %pd = ttg.convert_layout %p : tensor<64x32xf16, #out_qk> -> tensor<64x32xf16, #p_a>
      %vd = ttg.convert_layout %v : tensor<32x128xf16, #row> -> tensor<32x128xf16, #v_b>
      %pv = tt.dot %pd, %vd, %apv : tensor<64x32xf16, #p_a> * tensor<32x128xf16, #v_b> -> tensor<64x128xf32, #out_pv>

      scf.yield %qk, %pv : tensor<64x32xf32, #out_qk>, tensor<64x128xf32, #out_pv>
    } {tt.num_stages = 2 : i32}
    tt.return %r#0, %r#1 : tensor<64x32xf32, #out_qk>, tensor<64x128xf32, #out_pv>
  }

  // ACCEL-LABEL: tt.func public @sqmma_col_source_a
  // ACCEL: %[[A_SMEM:.+]] = ttg.local_alloc {{.*}} {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = false}
  // ACCEL: ttmg.squad_dot %[[A_SMEM]], {{.*}} {
  // ACCEL-SAME: layoutA = 1 : i32
  // BF16-ACCEL-LABEL: tt.func public @sqmma_col_source_a
  // BF16-ACCEL: %[[BF16_A_SMEM:.+]] = ttg.local_alloc {{.*}} {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = false}
  // BF16-ACCEL: ttmg.squad_dot %[[BF16_A_SMEM]], {{.*}} {
  // BF16-ACCEL-SAME: layoutA = 1 : i32
  tt.func public @sqmma_col_source_a(%a: tensor<64x128xf16, #col>, %b: tensor<128x32xf16, #row>, %acc: tensor<64x32xf32, #out_qk>) -> tensor<64x32xf32, #out_qk> {
    %ad = ttg.convert_layout %a : tensor<64x128xf16, #col> -> tensor<64x128xf16, #q_a>
    %bd = ttg.convert_layout %b : tensor<128x32xf16, #row> -> tensor<128x32xf16, #k_b>
    %dot = tt.dot %ad, %bd, %acc : tensor<64x128xf16, #q_a> * tensor<128x32xf16, #k_b> -> tensor<64x32xf32, #out_qk>
    tt.return %dot : tensor<64x32xf32, #out_qk>
  }

  // Rank-3 operands retain the established canonical row-major plan even when
  // the source distributed layouts prefer a different memory order.
  // ACCEL-LABEL: tt.func public @sqmma_rank3_canonical_order
  // ACCEL-DAG: %[[A3_SMEM:.+]] = ttg.local_alloc {{.*}} {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // ACCEL-DAG: %[[B3_SMEM:.+]] = ttg.local_alloc {{.*}} {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true}
  // ACCEL: ttmg.squad_dot %[[A3_SMEM]], %[[B3_SMEM]], {{.*}} {
  // ACCEL-SAME: layoutA = 0 : i32
  // ACCEL-SAME: layoutB = 0 : i32
  // BF16-ACCEL-LABEL: tt.func public @sqmma_rank3_canonical_order
  // BF16-ACCEL-DAG: %[[BF16_A3_SMEM:.+]] = ttg.local_alloc {{.*}} {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // BF16-ACCEL-DAG: %[[BF16_B3_SMEM:.+]] = ttg.local_alloc {{.*}} {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true}
  // BF16-ACCEL: ttmg.squad_dot %[[BF16_A3_SMEM]], %[[BF16_B3_SMEM]], {{.*}} {
  // BF16-ACCEL-SAME: layoutA = 0 : i32
  // BF16-ACCEL-SAME: layoutB = 0 : i32
  tt.func public @sqmma_rank3_canonical_order(%a: tensor<2x64x64xf16, #a_3d_src>, %b: tensor<2x64x64xf16, #b_3d_src>, %acc: tensor<2x64x64xf32, #out_3d>) -> tensor<2x64x64xf32, #out_3d> {
    %ad = ttg.convert_layout %a : tensor<2x64x64xf16, #a_3d_src> -> tensor<2x64x64xf16, #a_3d>
    %bd = ttg.convert_layout %b : tensor<2x64x64xf16, #b_3d_src> -> tensor<2x64x64xf16, #b_3d>
    %dot = tt.dot %ad, %bd, %acc : tensor<2x64x64xf16, #a_3d> * tensor<2x64x64xf16, #b_3d> -> tensor<2x64x64xf32, #out_3d>
    tt.return %dot : tensor<2x64x64xf32, #out_3d>
  }
}
