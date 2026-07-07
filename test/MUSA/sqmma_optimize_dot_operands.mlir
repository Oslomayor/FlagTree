// RUN: triton-opt %s --tritonmusa-optimize-dot-operands -canonicalize | FileCheck %s

#blocked_a = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked_r = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_rt = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked_flat = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked_matrix = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear_tr_reshape = #ttg.linear<{register = [[2, 0], [4, 0]], lane = [[8, 0], [16, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 8], [1, 0]], block = []}>
#mma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 16]}>
#mma_flat = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 32]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#shared_flat_a = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 16, order = [1, 0]}>
#shared_flat_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#shared_reshape_b = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 16], [8, 0]]}, alignment = 16>
#shared_nested_a = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 8], [16, 0]]}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @sqmma_plain_trans_local_alloc
  // CHECK-NOT: tt.trans
  // CHECK: %[[A_ALLOC:.+]] = ttg.local_alloc %arg0 {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: %[[B_ALLOC:.+]] = ttg.local_alloc %arg0 {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false}
  // CHECK: %[[B_VIEW:.+]] = ttg.memdesc_trans %[[B_ALLOC]] {order = array<i32: 1, 0>, sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false}
  // CHECK: ttmg.squad_dot %[[A_ALLOC]], %[[B_VIEW]]
  tt.func public @sqmma_plain_trans_local_alloc(%a: tensor<32x16xf16, #blocked_a>, %acc: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma> {
    %false = arith.constant false
    %b = tt.trans %a {order = array<i32: 1, 0>} : tensor<32x16xf16, #blocked_a> -> tensor<16x32xf16, #blocked_t>
    %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    %b_smem = ttg.local_alloc %b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false} : (tensor<16x32xf16, #blocked_t>) -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    %dot = ttmg.squad_dot %a_smem, %b_smem, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 1 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
    tt.return %dot : tensor<32x32xf32, #mma>
  }

  // CHECK-LABEL: tt.func public @sqmma_plain_reshape_local_alloc
  // CHECK-NOT: tt.reshape
  // CHECK: %[[RA_ALLOC:.+]] = ttg.local_alloc %arg0 {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: %[[RB_ALLOC:.+]] = ttg.local_alloc %arg0 {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true}
  // CHECK: %[[RB_VIEW:.+]] = ttg.memdesc_reshape %[[RB_ALLOC]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true}
  // CHECK: ttmg.squad_dot %[[RA_ALLOC]], %[[RB_VIEW]]
  tt.func public @sqmma_plain_reshape_local_alloc(%a: tensor<32x16xf16, #blocked_a>, %acc: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma> {
    %false = arith.constant false
    %b = tt.reshape %a : tensor<32x16xf16, #blocked_a> -> tensor<16x32xf16, #blocked_r>
    %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    %b_smem = ttg.local_alloc %b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<16x32xf16, #blocked_r>) -> !ttg.memdesc<16x32xf16, #shared_reshape_b, #smem, mutable>
    %dot = ttmg.squad_dot %a_smem, %b_smem, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_reshape_b, #smem, mutable> -> tensor<32x32xf32, #mma>
    tt.return %dot : tensor<32x32xf32, #mma>
  }

  // CHECK-LABEL: tt.func public @sqmma_nested_view_local_alloc
  // CHECK-NOT: tt.reshape
  // CHECK-NOT: tt.trans
  // CHECK: %[[B_ALLOC:.+]] = ttg.local_alloc %arg0 {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: %[[B_RESHAPE:.+]] = ttg.memdesc_reshape %[[B_ALLOC]]
  // CHECK: %[[B_VIEW:.+]] = ttg.memdesc_trans %[[B_RESHAPE]] {order = array<i32: 1, 0>, sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: ttmg.squad_dot %[[B_VIEW]]
  tt.func public @sqmma_nested_view_local_alloc(%a: tensor<32x16xf16, #blocked_a>, %b: !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>, %acc: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma> {
    %false = arith.constant false
    %r = tt.reshape %a : tensor<32x16xf16, #blocked_a> -> tensor<16x32xf16, #blocked_r>
    %t = tt.trans %r {order = array<i32: 1, 0>} : tensor<16x32xf16, #blocked_r> -> tensor<32x16xf16, #blocked_rt>
    %a_smem = ttg.local_alloc %t {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_rt>) -> !ttg.memdesc<32x16xf16, #shared_nested_a, #smem, mutable>
    %dot = ttmg.squad_dot %a_smem, %b, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_nested_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
    tt.return %dot : tensor<32x32xf32, #mma>
  }

  // CHECK-LABEL: tt.func public @sqmma_trans_reshape_view_local_alloc
  // CHECK-NOT: tt.trans
  // CHECK-NOT: tt.reshape
  // CHECK: %[[B_ALLOC:.+]] = ttg.local_alloc %arg0 {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: %[[B_TRANS:.+]] = ttg.memdesc_trans %[[B_ALLOC]] {order = array<i32: 1, 0>, sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: %[[B_VIEW:.+]] = ttg.memdesc_reshape %[[B_TRANS]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: ttmg.squad_dot %[[B_VIEW]]
  tt.func public @sqmma_trans_reshape_view_local_alloc(%a: tensor<32x16xf16, #blocked_a>, %b: !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>, %acc: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma> {
    %false = arith.constant false
    %t = tt.trans %a {order = array<i32: 1, 0>} : tensor<32x16xf16, #blocked_a> -> tensor<16x32xf16, #blocked_t>
    %r = tt.reshape %t : tensor<16x32xf16, #blocked_t> -> tensor<32x16xf16, #linear_tr_reshape>
    %a_smem = ttg.local_alloc %r {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #linear_tr_reshape>) -> !ttg.memdesc<32x16xf16, #shared_nested_a, #smem, mutable>
    %dot = ttmg.squad_dot %a_smem, %b, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_nested_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
    tt.return %dot : tensor<32x32xf32, #mma>
  }

  // CHECK-LABEL: tt.func public @sqmma_plain_flat_reshape_keeps_swizzled_alloc
  // CHECK: %[[A_RESHAPE:.+]] = tt.reshape %arg0
  // CHECK: %[[B_RESHAPE:.+]] = tt.reshape %arg1
  // CHECK: %[[A_ALLOC:.+]] = ttg.local_alloc %[[A_RESHAPE]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: %[[B_ALLOC:.+]] = ttg.local_alloc %[[B_RESHAPE]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true}
  // CHECK-NOT: ttg.memdesc_reinterpret
  // CHECK: ttmg.squad_dot %[[A_ALLOC]], %[[B_ALLOC]]
  tt.func public @sqmma_plain_flat_reshape_keeps_swizzled_alloc(%a: tensor<1024xbf16, #blocked_flat>, %b: tensor<1024xbf16, #blocked_flat>, %acc: tensor<32x32xf32, #mma_flat>) -> tensor<32x32xf32, #mma_flat> {
    %false = arith.constant false
    %a_r = tt.reshape %a : tensor<1024xbf16, #blocked_flat> -> tensor<32x32xbf16, #blocked_matrix>
    %b_r = tt.reshape %b : tensor<1024xbf16, #blocked_flat> -> tensor<32x32xbf16, #blocked_matrix>
    %a_smem = ttg.local_alloc %a_r {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x32xbf16, #blocked_matrix>) -> !ttg.memdesc<32x32xbf16, #shared_flat_a, #smem, mutable>
    %b_smem = ttg.local_alloc %b_r {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<32x32xbf16, #blocked_matrix>) -> !ttg.memdesc<32x32xbf16, #shared_flat_b, #smem, mutable>
    %dot = ttmg.squad_dot %a_smem, %b_smem, %acc, %false {eltTypeA = 5 : i32, eltTypeB = 5 : i32, eltTypeC = 7 : i32, k = 32 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x32xbf16, #shared_flat_a, #smem, mutable> * !ttg.memdesc<32x32xbf16, #shared_flat_b, #smem, mutable> -> tensor<32x32xf32, #mma_flat>
    tt.return %dot : tensor<32x32xf32, #mma_flat>
  }
}
