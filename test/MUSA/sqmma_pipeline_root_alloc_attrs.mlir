// RUN: triton-opt %s --tritonmusa-pipeline=num-stages=3 -canonicalize | FileCheck %s

#blocked_a = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked_r = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_rt = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear_tr_reshape = #ttg.linear<{register = [[2, 0], [4, 0]], lane = [[8, 0], [16, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 8], [1, 0]], block = []}>
#mma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 16]}>
#mma_i8 = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 32]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#shared_bt = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [0, 1]}>
#shared_plain = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_reshape_b = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 16], [8, 0]]}, alignment = 16>
#shared_nested_a = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 8], [16, 0]]}, alignment = 16>
#shared_linear_plain = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]]}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @sqmma_pipeline_root_alloc_attrs
  // CHECK: %[[ROOT:.+]] = ttg.local_alloc {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : () -> !ttg.memdesc<3x32x16xf16, #shared{{[0-9]*}}, #smem, mutable>
  // CHECK: scf.for
  // CHECK: ttg.memdesc_index %[[ROOT]]
  tt.func public @sqmma_pipeline_root_alloc_attrs(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %trip_count: index) -> tensor<32x32xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_b = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #blocked_b>
    %false = arith.constant false
    %a_step = arith.constant dense<16> : tensor<32x16xi32, #blocked_a>
    %offs_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_k_exp = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x16xi32, #blocked_a>
    %offs = tt.broadcast %offs_k_exp : tensor<1x16xi32, #blocked_a> -> tensor<32x16xi32, #blocked_a>
    %a_splat = tt.splat %A : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked_a>
    %a_ptr = tt.addptr %a_splat, %offs : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
    %b_smem = ttg.local_alloc %cst_b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<16x32xf16, #blocked_b>) -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    %result:2 = scf.for %iv = %c0 to %trip_count step %c1 iter_args(%a_iter = %a_ptr, %acc_iter = %cst_acc) -> (tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>) {
      %a = tt.load %a_iter : tensor<32x16x!tt.ptr<f16>, #blocked_a>
      %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
      %dot = ttmg.squad_dot %a_smem, %b_smem, %acc_iter, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
      %next_a = tt.addptr %a_iter, %a_step : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
      scf.yield %next_a, %dot : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>
    } {tt.num_stages = 3 : i32}
    ttg.local_dealloc %b_smem : !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    tt.return %result#1 : tensor<32x32xf32, #mma>
  }

  // This is a small-copy pipeline fallback case.  The SQMMA landing source
  // encoding inferred through the tensor transpose only provides a 2-byte f16
  // copy vector, below the 4-byte async-copy minimum, so this pass must keep
  // the tensor view + local_alloc path.
  // CHECK-LABEL: tt.func public @sqmma_pipeline_transpose_operand_memdesc
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: scf.for
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: %[[B_VIEW:.+]] = tt.trans
  // CHECK: %[[A_SMEM:.+]] = ttg.local_alloc %{{.*}} {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: %[[B_SMEM:.+]] = ttg.local_alloc %[[B_VIEW]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false}
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: ttmg.squad_dot %[[A_SMEM]], %[[B_SMEM]]
  tt.func public @sqmma_pipeline_transpose_operand_memdesc(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %trip_count: index) -> tensor<32x32xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %false = arith.constant false
    %a_step = arith.constant dense<16> : tensor<32x16xi32, #blocked_a>
    %offs_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_k_exp = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x16xi32, #blocked_a>
    %offs = tt.broadcast %offs_k_exp : tensor<1x16xi32, #blocked_a> -> tensor<32x16xi32, #blocked_a>
    %a_splat = tt.splat %A : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked_a>
    %a_ptr = tt.addptr %a_splat, %offs : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
    %result:2 = scf.for %iv = %c0 to %trip_count step %c1 iter_args(%a_iter = %a_ptr, %acc_iter = %cst_acc) -> (tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>) {
      %a = tt.load %a_iter : tensor<32x16x!tt.ptr<f16>, #blocked_a>
      %b = tt.trans %a {order = array<i32: 1, 0>} : tensor<32x16xf16, #blocked_a> -> tensor<16x32xf16, #blocked_b>
      %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
      %b_smem = ttg.local_alloc %b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false} : (tensor<16x32xf16, #blocked_b>) -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
      %dot = ttmg.squad_dot %a_smem, %b_smem, %acc_iter, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 1 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
      %next_a = tt.addptr %a_iter, %a_step : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
      scf.yield %next_a, %dot : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>
    } {tt.num_stages = 3 : i32}
    tt.return %result#1 : tensor<32x32xf32, #mma>
  }

  // CHECK-LABEL: tt.func public @sqmma_pipeline_mixed_reshape_operand_preserves_target
  // CHECK: ttg.async_wait
  // CHECK-NOT: ttg.local_load
  // CHECK: %[[A_VIEW:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: %[[B_SOURCE:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
  // CHECK: %[[B_VIEW:.+]] = ttg.memdesc_reshape %[[B_SOURCE]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true}
  // CHECK-NOT: ttg.local_load
  // CHECK: ttmg.squad_dot %[[A_VIEW]], %[[B_VIEW]]
  tt.func public @sqmma_pipeline_mixed_reshape_operand_preserves_target(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %trip_count: index) -> tensor<32x32xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %false = arith.constant false
    %a_step = arith.constant dense<16> : tensor<32x16xi32, #blocked_a>
    %offs_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_k_exp = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x16xi32, #blocked_a>
    %offs = tt.broadcast %offs_k_exp : tensor<1x16xi32, #blocked_a> -> tensor<32x16xi32, #blocked_a>
    %a_splat = tt.splat %A : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked_a>
    %a_ptr = tt.addptr %a_splat, %offs : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
    %result:2 = scf.for %iv = %c0 to %trip_count step %c1 iter_args(%a_iter = %a_ptr, %acc_iter = %cst_acc) -> (tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>) {
      %a = tt.load %a_iter : tensor<32x16x!tt.ptr<f16>, #blocked_a>
      %b = tt.reshape %a : tensor<32x16xf16, #blocked_a> -> tensor<16x32xf16, #blocked_r>
      %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
      %b_smem = ttg.local_alloc %b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<16x32xf16, #blocked_r>) -> !ttg.memdesc<16x32xf16, #shared_reshape_b, #smem, mutable>
      %dot = ttmg.squad_dot %a_smem, %b_smem, %acc_iter, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_reshape_b, #smem, mutable> -> tensor<32x32xf32, #mma>
      %next_a = tt.addptr %a_iter, %a_step : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
      scf.yield %next_a, %dot : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>
    } {tt.num_stages = 3 : i32}
    tt.return %result#1 : tensor<32x32xf32, #mma>
  }

  // CHECK-LABEL: tt.func public @sqmma_pipeline_view_only_transpose_operand_memdesc
  // CHECK: %[[ROOT:.+]] = ttg.local_alloc {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false} : () -> !ttg.memdesc<3x32x16xf16, #shared{{[0-9]*}}, #smem, mutable>
  // CHECK: ttg.async_wait
  // CHECK-NOT: ttg.local_load
  // CHECK: %[[B_SOURCE:.+]] = ttg.memdesc_index %[[ROOT]][{{.*}}]
  // CHECK: %[[B_VIEW:.+]] = ttg.memdesc_trans %[[B_SOURCE]] {order = array<i32: 1, 0>, sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false}
  // CHECK-NOT: ttg.local_load
  // CHECK: ttmg.squad_dot %{{.*}}, %[[B_VIEW]]
  tt.func public @sqmma_pipeline_view_only_transpose_operand_memdesc(%B: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %trip_count: index) -> tensor<32x32xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_a = arith.constant dense<0.000000e+00> : tensor<32x16xf16, #blocked_a>
    %false = arith.constant false
    %b_step = arith.constant dense<16> : tensor<32x16xi32, #blocked_a>
    %offs_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_k_exp = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x16xi32, #blocked_a>
    %offs = tt.broadcast %offs_k_exp : tensor<1x16xi32, #blocked_a> -> tensor<32x16xi32, #blocked_a>
    %b_splat = tt.splat %B : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked_a>
    %b_ptr = tt.addptr %b_splat, %offs : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
    %a_smem = ttg.local_alloc %cst_a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    %result:2 = scf.for %iv = %c0 to %trip_count step %c1 iter_args(%b_iter = %b_ptr, %acc_iter = %cst_acc) -> (tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>) {
      %b_raw = tt.load %b_iter : tensor<32x16x!tt.ptr<f16>, #blocked_a>
      %b = tt.trans %b_raw {order = array<i32: 1, 0>} : tensor<32x16xf16, #blocked_a> -> tensor<16x32xf16, #blocked_b>
      %b_smem = ttg.local_alloc %b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = false} : (tensor<16x32xf16, #blocked_b>) -> !ttg.memdesc<16x32xf16, #shared_bt, #smem, mutable>
      %dot = ttmg.squad_dot %a_smem, %b_smem, %acc_iter, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 1 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_bt, #smem, mutable> -> tensor<32x32xf32, #mma>
      %next_b = tt.addptr %b_iter, %b_step : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
      scf.yield %next_b, %dot : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>
    } {tt.num_stages = 3 : i32}
    ttg.local_dealloc %a_smem : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    tt.return %result#1 : tensor<32x32xf32, #mma>
  }

  // Same small-copy fallback coverage for a nested reshape + transpose chain.
  // The optimized memdesc-view contract is covered by optimize-dot-operands;
  // this pipeline-only run intentionally exercises the register-pipeline path.
  // CHECK-LABEL: tt.func public @sqmma_pipeline_nested_view_operand_memdesc
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: scf.for
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: %[[A_RESHAPE:.+]] = tt.reshape
  // CHECK: %[[A_VIEW:.+]] = tt.trans %[[A_RESHAPE]]
  // CHECK: %[[A_SMEM:.+]] = ttg.local_alloc %[[A_VIEW]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: ttmg.squad_dot %[[A_SMEM]]
  tt.func public @sqmma_pipeline_nested_view_operand_memdesc(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %trip_count: index) -> tensor<32x32xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_b = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #blocked_b>
    %false = arith.constant false
    %a_step = arith.constant dense<16> : tensor<32x16xi32, #blocked_a>
    %offs_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_k_exp = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x16xi32, #blocked_a>
    %offs = tt.broadcast %offs_k_exp : tensor<1x16xi32, #blocked_a> -> tensor<32x16xi32, #blocked_a>
    %a_splat = tt.splat %A : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked_a>
    %a_ptr = tt.addptr %a_splat, %offs : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
    %b_smem = ttg.local_alloc %cst_b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<16x32xf16, #blocked_b>) -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    %result:2 = scf.for %iv = %c0 to %trip_count step %c1 iter_args(%a_iter = %a_ptr, %acc_iter = %cst_acc) -> (tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>) {
      %a_raw = tt.load %a_iter : tensor<32x16x!tt.ptr<f16>, #blocked_a>
      %a_reshape = tt.reshape %a_raw : tensor<32x16xf16, #blocked_a> -> tensor<16x32xf16, #blocked_r>
      %a_view = tt.trans %a_reshape {order = array<i32: 1, 0>} : tensor<16x32xf16, #blocked_r> -> tensor<32x16xf16, #blocked_rt>
      %a_smem = ttg.local_alloc %a_view {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_rt>) -> !ttg.memdesc<32x16xf16, #shared_nested_a, #smem, mutable>
      %dot = ttmg.squad_dot %a_smem, %b_smem, %acc_iter, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_nested_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
      %next_a = tt.addptr %a_iter, %a_step : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
      scf.yield %next_a, %dot : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>
    } {tt.num_stages = 3 : i32}
    ttg.local_dealloc %b_smem : !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    tt.return %result#1 : tensor<32x32xf32, #mma>
  }

  // CHECK-LABEL: tt.func public @sqmma_pipeline_residual_non_sqmma_local_alloc
  // CHECK: %[[ROOT:.+]] = ttg.local_alloc {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : () -> !ttg.memdesc<3x32x16xf16, #shared{{[0-9]*}}, #smem, mutable>
  // CHECK: ttg.async_wait
  // CHECK: %[[LOCAL:.+]] = ttg.local_load %{{.*}} : !ttg.memdesc<32x16xf16, #shared{{[0-9]*}}, #smem, mutable> -> tensor<32x16xf16, #blocked{{[0-9]*}}>
  // CHECK: ttmg.squad_dot
  // CHECK: scf.yield {{.*}}, %[[LOCAL]]
  tt.func public @sqmma_pipeline_residual_non_sqmma_local_alloc(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %trip_count: index) -> (tensor<32x32xf32, #mma>, tensor<32x16xf16, #blocked_a>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_aux = arith.constant dense<0.000000e+00> : tensor<32x16xf16, #blocked_a>
    %cst_b = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #blocked_b>
    %false = arith.constant false
    %a_step = arith.constant dense<16> : tensor<32x16xi32, #blocked_a>
    %offs_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_k_exp = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x16xi32, #blocked_a>
    %offs = tt.broadcast %offs_k_exp : tensor<1x16xi32, #blocked_a> -> tensor<32x16xi32, #blocked_a>
    %a_splat = tt.splat %A : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked_a>
    %a_ptr = tt.addptr %a_splat, %offs : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
    %b_smem = ttg.local_alloc %cst_b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<16x32xf16, #blocked_b>) -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    %result:3 = scf.for %iv = %c0 to %trip_count step %c1 iter_args(%a_iter = %a_ptr, %acc_iter = %cst_acc, %aux_iter = %cst_aux) -> (tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>, tensor<32x16xf16, #blocked_a>) {
      %a = tt.load %a_iter : tensor<32x16x!tt.ptr<f16>, #blocked_a>
      %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
      %a_aux = ttg.local_alloc %a : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
      %a_aux_load = ttg.local_load %a_aux : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> -> tensor<32x16xf16, #blocked_a>
      %dot = ttmg.squad_dot %a_smem, %b_smem, %acc_iter, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
      %next_a = tt.addptr %a_iter, %a_step : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
      scf.yield %next_a, %dot, %a_aux_load : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>, tensor<32x16xf16, #blocked_a>
    } {tt.num_stages = 3 : i32}
    ttg.local_dealloc %b_smem : !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    tt.return %result#1, %result#2 : tensor<32x32xf32, #mma>, tensor<32x16xf16, #blocked_a>
  }

  // CHECK-LABEL: tt.func public @sqmma_pipeline_same_alloc_residual_local_load
  // CHECK: ttg.async_wait
  // CHECK: %[[SQMMA_VIEW:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: %[[LOCAL:.+]] = ttg.local_load %[[SQMMA_VIEW]]
  // CHECK: ttmg.squad_dot %[[SQMMA_VIEW]]
  // CHECK: scf.yield {{.*}}, %[[LOCAL]]
  tt.func public @sqmma_pipeline_same_alloc_residual_local_load(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %trip_count: index) -> (tensor<32x32xf32, #mma>, tensor<32x16xf16, #blocked_a>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_aux = arith.constant dense<0.000000e+00> : tensor<32x16xf16, #blocked_a>
    %cst_b = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #blocked_b>
    %false = arith.constant false
    %a_step = arith.constant dense<16> : tensor<32x16xi32, #blocked_a>
    %offs_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_k_exp = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x16xi32, #blocked_a>
    %offs = tt.broadcast %offs_k_exp : tensor<1x16xi32, #blocked_a> -> tensor<32x16xi32, #blocked_a>
    %a_splat = tt.splat %A : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked_a>
    %a_ptr = tt.addptr %a_splat, %offs : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
    %b_smem = ttg.local_alloc %cst_b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<16x32xf16, #blocked_b>) -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    %result:3 = scf.for %iv = %c0 to %trip_count step %c1 iter_args(%a_iter = %a_ptr, %acc_iter = %cst_acc, %aux_iter = %cst_aux) -> (tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>, tensor<32x16xf16, #blocked_a>) {
      %a = tt.load %a_iter : tensor<32x16x!tt.ptr<f16>, #blocked_a>
      %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
      %a_aux_load = ttg.local_load %a_smem : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> -> tensor<32x16xf16, #blocked_a>
      %dot = ttmg.squad_dot %a_smem, %b_smem, %acc_iter, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
      %next_a = tt.addptr %a_iter, %a_step : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
      scf.yield %next_a, %dot, %a_aux_load : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>, tensor<32x16xf16, #blocked_a>
    } {tt.num_stages = 3 : i32}
    ttg.local_dealloc %b_smem : !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    tt.return %result#1, %result#2 : tensor<32x32xf32, #mma>, tensor<32x16xf16, #blocked_a>
  }

  // CHECK-LABEL: tt.func public @sqmma_pipeline_layout_equivalent_source_reinterpret
  // CHECK: %[[ROOT:.+]] = ttg.local_alloc {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : () -> !ttg.memdesc<3x32x16xf16, #shared{{[0-9]*}}, #smem, mutable>
  // CHECK: ttg.async_wait
  // CHECK: %[[SOURCE:.+]] = ttg.memdesc_index %[[ROOT]][{{.*}}] : !ttg.memdesc<3x32x16xf16, #shared{{[0-9]*}}, #smem, mutable> -> !ttg.memdesc<32x16xf16, #shared{{[0-9]*}}, #smem, mutable>
  // CHECK: %[[REINTERP:.+]] = ttg.memdesc_reinterpret %[[SOURCE]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: ttmg.squad_dot %[[REINTERP]]
  tt.func public @sqmma_pipeline_layout_equivalent_source_reinterpret(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %trip_count: index) -> tensor<32x32xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_b = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #blocked_b>
    %false = arith.constant false
    %a_step = arith.constant dense<16> : tensor<32x16xi32, #blocked_a>
    %offs_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_k_exp = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x16xi32, #blocked_a>
    %offs = tt.broadcast %offs_k_exp : tensor<1x16xi32, #blocked_a> -> tensor<32x16xi32, #blocked_a>
    %a_splat = tt.splat %A : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked_a>
    %a_ptr = tt.addptr %a_splat, %offs : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
    %b_smem = ttg.local_alloc %cst_b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<16x32xf16, #blocked_b>) -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    %result:2 = scf.for %iv = %c0 to %trip_count step %c1 iter_args(%a_iter = %a_ptr, %acc_iter = %cst_acc) -> (tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>) {
      %a = tt.load %a_iter : tensor<32x16x!tt.ptr<f16>, #blocked_a>
      %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_plain, #smem, mutable>
      %a_linear = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_linear_plain, #smem, mutable>
      %dot0 = ttmg.squad_dot %a_linear, %b_smem, %acc_iter, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_linear_plain, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
      %dot1 = ttmg.squad_dot %a_smem, %b_smem, %dot0, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_plain, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
      %next_a = tt.addptr %a_iter, %a_step : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
      scf.yield %next_a, %dot1 : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>
    } {tt.num_stages = 3 : i32}
    ttg.local_dealloc %b_smem : !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    tt.return %result#1 : tensor<32x32xf32, #mma>
  }

  // Same small-copy fallback coverage for a transpose + reshape chain.  The
  // inferred SQMMA landing source has copyVecBytes < 4, so no async copy is
  // expected in this pipeline-only test.
  // CHECK-LABEL: tt.func public @sqmma_pipeline_trans_reshape_operand_memdesc
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: scf.for
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: %[[A_TRANS:.+]] = tt.trans
  // CHECK: %[[A_VIEW:.+]] = tt.reshape %[[A_TRANS]]
  // CHECK: %[[A_SMEM:.+]] = ttg.local_alloc %[[A_VIEW]] {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: ttmg.squad_dot %[[A_SMEM]]
  tt.func public @sqmma_pipeline_trans_reshape_operand_memdesc(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %trip_count: index) -> tensor<32x32xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_b = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #blocked_b>
    %false = arith.constant false
    %a_step = arith.constant dense<16> : tensor<32x16xi32, #blocked_a>
    %offs_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_k_exp = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x16xi32, #blocked_a>
    %offs = tt.broadcast %offs_k_exp : tensor<1x16xi32, #blocked_a> -> tensor<32x16xi32, #blocked_a>
    %a_splat = tt.splat %A : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked_a>
    %a_ptr = tt.addptr %a_splat, %offs : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
    %b_smem = ttg.local_alloc %cst_b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<16x32xf16, #blocked_b>) -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    %result:2 = scf.for %iv = %c0 to %trip_count step %c1 iter_args(%a_iter = %a_ptr, %acc_iter = %cst_acc) -> (tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>) {
      %a_raw = tt.load %a_iter : tensor<32x16x!tt.ptr<f16>, #blocked_a>
      %a_trans = tt.trans %a_raw {order = array<i32: 1, 0>} : tensor<32x16xf16, #blocked_a> -> tensor<16x32xf16, #blocked_b>
      %a_view = tt.reshape %a_trans : tensor<16x32xf16, #blocked_b> -> tensor<32x16xf16, #linear_tr_reshape>
      %a_smem = ttg.local_alloc %a_view {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #linear_tr_reshape>) -> !ttg.memdesc<32x16xf16, #shared_nested_a, #smem, mutable>
      %dot = ttmg.squad_dot %a_smem, %b_smem, %acc_iter, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_nested_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
      %next_a = tt.addptr %a_iter, %a_step : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x16xi32, #blocked_a>
      scf.yield %next_a, %dot : tensor<32x16x!tt.ptr<f16>, #blocked_a>, tensor<32x32xf32, #mma>
    } {tt.num_stages = 3 : i32}
    ttg.local_dealloc %b_smem : !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    tt.return %result#1 : tensor<32x32xf32, #mma>
  }

  // CHECK-LABEL: tt.func public @sqmma_pipeline_shared_root_conflicting_final_contracts
  // CHECK: %[[ROOT:.+]] = ttg.local_alloc : () -> !ttg.memdesc<3x32x32xi8, #shared{{[0-9]*}}, #smem, mutable>
  // CHECK-NOT: ttg.local_alloc
  // CHECK: scf.for
  // CHECK-DAG: %[[B_VIEW:.+]] = ttg.memdesc_index %[[ROOT]][{{.*}}] {sqmma.elem_bytes = 1 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true}
  // CHECK-DAG: %[[A_VIEW:.+]] = ttg.memdesc_index %[[ROOT]][{{.*}}] {sqmma.elem_bytes = 1 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true}
  // CHECK: ttmg.squad_dot %[[A_VIEW]], %[[B_VIEW]]
  tt.func public @sqmma_pipeline_shared_root_conflicting_final_contracts(%A: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %trip_count: index) -> tensor<32x32xi32, #mma_i8> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_acc = arith.constant dense<0> : tensor<32x32xi32, #mma_i8>
    %false = arith.constant false
    %a_step = arith.constant dense<32> : tensor<32x32xi32, #blocked_a>
    %offs_k = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_k_exp = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x32xi32, #blocked_a>
    %offs = tt.broadcast %offs_k_exp : tensor<1x32xi32, #blocked_a> -> tensor<32x32xi32, #blocked_a>
    %a_splat = tt.splat %A : !tt.ptr<i8> -> tensor<32x32x!tt.ptr<i8>, #blocked_a>
    %a_ptr = tt.addptr %a_splat, %offs : tensor<32x32x!tt.ptr<i8>, #blocked_a>, tensor<32x32xi32, #blocked_a>
    %result:2 = scf.for %iv = %c0 to %trip_count step %c1 iter_args(%a_iter = %a_ptr, %acc_iter = %cst_acc) -> (tensor<32x32x!tt.ptr<i8>, #blocked_a>, tensor<32x32xi32, #mma_i8>) {
      %a = tt.load %a_iter : tensor<32x32x!tt.ptr<i8>, #blocked_a>
      %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 1 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x32xi8, #blocked_a>) -> !ttg.memdesc<32x32xi8, #shared_a, #smem, mutable>
      %b_smem = ttg.local_alloc %a {sqmma.elem_bytes = 1 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<32x32xi8, #blocked_a>) -> !ttg.memdesc<32x32xi8, #shared_a, #smem, mutable>
      %dot = ttmg.squad_dot %a_smem, %b_smem, %acc_iter, %false {eltTypeA = 0 : i32, eltTypeB = 0 : i32, eltTypeC = 1 : i32, inputPrecision = 0 : i32, k = 32 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x32xi8, #shared_a, #smem, mutable> * !ttg.memdesc<32x32xi8, #shared_a, #smem, mutable> -> tensor<32x32xi32, #mma_i8>
      %next_a = tt.addptr %a_iter, %a_step : tensor<32x32x!tt.ptr<i8>, #blocked_a>, tensor<32x32xi32, #blocked_a>
      scf.yield %next_a, %dot : tensor<32x32x!tt.ptr<i8>, #blocked_a>, tensor<32x32xi32, #mma_i8>
    } {tt.num_stages = 3 : i32}
    tt.return %result#1 : tensor<32x32xi32, #mma_i8>
  }
}
