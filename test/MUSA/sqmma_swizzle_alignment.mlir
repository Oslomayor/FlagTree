// RUN: triton-opt %s --allocate-musa-shared-memory --convert-triton-musagpu-to-llvm=compute-capability=31 -reconcile-unrealized-casts | FileCheck %s

#mma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 16]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sqmma_alignment_from_final_view(%acc: tensor<32x32xf32, #mma>) {
    %false = arith.constant false
    %a_root = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    %b_root = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    %a_view = ttg.memdesc_reinterpret %a_root {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    %b_view = ttg.memdesc_reinterpret %b_root {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 1 : i32, sqmma.row_major = true} : !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    %dot = ttmg.squad_dot %a_view, %b_view, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma>
    ttg.local_dealloc %a_root : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    ttg.local_dealloc %b_root : !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>
    tt.return
  }
}

// CHECK: llvm.mlir.global external @global_smem() {{.*}}alignment = 4096 : i64
// CHECK-LABEL: llvm.func @sqmma_alignment_from_final_view
