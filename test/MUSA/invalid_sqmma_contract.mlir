// RUN: triton-opt %s -split-input-file -verify-diagnostics

#blocked_a = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 16]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sqmma_operand_contract_op_idx_mismatch(%a: tensor<32x16xf16, #blocked_a>, %b_smem: !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>, %acc: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma> {
    %false = arith.constant false
    %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    // expected-error@+1 {{SQMMA operand A producer sqmma.op_idx must match the operand index}}
    %dot = ttmg.squad_dot %a_smem, %b_smem, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
    tt.return %dot : tensor<32x32xf32, #mma>
  }
}

// -----

#blocked_a = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 16]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sqmma_operand_contract_elem_bytes_mismatch(%a: tensor<32x16xf16, #blocked_a>, %b_smem: !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>, %acc: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma> {
    %false = arith.constant false
    %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 4 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    // expected-error@+1 {{SQMMA operand A producer sqmma.elem_bytes must match the memdesc element type}}
    %dot = ttmg.squad_dot %a_smem, %b_smem, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
    tt.return %dot : tensor<32x32xf32, #mma>
  }
}

// -----

#blocked_a = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 16]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sqmma_operand_contract_row_major_mismatch(%a: tensor<32x16xf16, #blocked_a>, %b_smem: !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>, %acc: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma> {
    %false = arith.constant false
    %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = false} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    // expected-error@+1 {{SQMMA operand A producer sqmma.row_major must match the consumer layout}}
    %dot = ttmg.squad_dot %a_smem, %b_smem, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
    tt.return %dot : tensor<32x32xf32, #mma>
  }
}

// -----

#blocked_a = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_r = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 16]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sqmma_operand_contract_view_producer_mismatch(%a: tensor<32x16xf16, #blocked_a>, %acc: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma> {
    %false = arith.constant false
    %a_smem = ttg.local_alloc %a {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    %b = tt.reshape %a : tensor<32x16xf16, #blocked_a> -> tensor<16x32xf16, #blocked_r>
    %b_smem = ttg.local_alloc %b {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<16x32xf16, #blocked_r>) -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    // expected-error@+1 {{SQMMA operand B producer sqmma.op_idx must match the operand index}}
    %dot = ttmg.squad_dot %a_smem, %b_smem, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
    tt.return %dot : tensor<32x32xf32, #mma>
  }
}
