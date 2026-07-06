// RUN: triton-opt %s -canonicalize | FileCheck %s --check-prefixes=TME,RESTAGE

#blocked_a = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_store = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_store = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 128, 32]}>
#mma_chain = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 32, 16]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 16, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 8, order = [1, 0]}>
#shared_store = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  // TME-LABEL: tt.func public @sqmma_final_tme_store_alloc_folds
  // TME-NOT: ttg.convert_layout
  // TME: %[[ALLOC:.+]] = ttg.local_alloc %arg0 : (tensor<32x1024xf16, #{{[A-Za-z0-9_]+}}>) -> !ttg.memdesc<32x1024xf16, #{{[A-Za-z0-9_]+}}, #smem>
  // TME: ttmg.async_tme_copy_local_to_global {{.*}}, %[[ALLOC]], {{.*}} : !tt.tensordesc<tensor<32x1024xf16, #{{[A-Za-z0-9_]+}}>>, !ttg.memdesc<32x1024xf16, #{{[A-Za-z0-9_]+}}, #smem>
  tt.func public @sqmma_final_tme_store_alloc_folds(%arg0: tensor<32x1024xf16, #mma_store>, %desc: !tt.tensordesc<tensor<32x1024xf16, #shared_store>>) {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.convert_layout %arg0 : tensor<32x1024xf16, #mma_store> -> tensor<32x1024xf16, #blocked_store>
    %alloc = ttg.local_alloc %0 : (tensor<32x1024xf16, #blocked_store>) -> !ttg.memdesc<32x1024xf16, #shared_store, #smem>
    ttmg.async_tme_copy_local_to_global %desc[%c0_i32, %c0_i32], %alloc, %true {blockShape = array<i32: 32, 1024>, cachePolicy = 0 : i32, innerPersistence = 2 : i32, outerPersistence = 2 : i32, swizzleGranularity = 0 : i32, swizzleLine = 1 : i32, swizzleStride = 3 : i32} : !tt.tensordesc<tensor<32x1024xf16, #shared_store>>, !ttg.memdesc<32x1024xf16, #shared_store, #smem>
    tt.return
  }

  // RESTAGE-LABEL: tt.func public @sqmma_chained_restaging_preserves_contract
  // RESTAGE-NOT: ttg.convert_layout
  // RESTAGE: %[[ALLOC:.+]] = ttg.local_alloc %arg0 {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #{{[A-Za-z0-9_]+}}>) -> !ttg.memdesc<32x16xf16, #{{[A-Za-z0-9_]+}}, #smem, mutable>
  // RESTAGE: ttmg.squad_dot %[[ALLOC]]
  tt.func public @sqmma_chained_restaging_preserves_contract(%arg0: tensor<32x16xf16, #mma_chain>, %b_smem: !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable>, %acc: tensor<32x32xf32, #mma_chain>) -> tensor<32x32xf32, #mma_chain> {
    %false = arith.constant false
    %0 = ttg.convert_layout %arg0 : tensor<32x16xf16, #mma_chain> -> tensor<32x16xf16, #blocked_a>
    %a_smem = ttg.local_alloc %0 {sqmma.elem_bytes = 2 : i32, sqmma.op_idx = 0 : i32, sqmma.row_major = true} : (tensor<32x16xf16, #blocked_a>) -> !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable>
    %dot = ttmg.squad_dot %a_smem, %b_smem, %acc, %false {eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, k = 16 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 32 : i32, n = 32 : i32} : !ttg.memdesc<32x16xf16, #shared_a, #smem, mutable> * !ttg.memdesc<16x32xf16, #shared_b, #smem, mutable> -> tensor<32x32xf32, #mma_chain>
    tt.return %dot : tensor<32x32xf32, #mma_chain>
  }
}
