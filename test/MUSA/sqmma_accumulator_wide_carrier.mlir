// RUN: triton-opt %s --convert-mtgpu-to-llvm=compute-capability=31 -reconcile-unrealized-casts | FileCheck %s

#mma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 128, 32]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @sqmma_pack_uses_wide_vector_carrier
  // CHECK: llvm.mlir.undef : vector<64xf32>
  // CHECK: builtin.unrealized_conversion_cast {{.*}} : vector<64xf32> to !mtgpu.sqmma_accumulator
  // CHECK-NOT: !llvm.struct<(vector<32xf32>
  tt.func public @sqmma_pack_uses_wide_vector_carrier(%arg0: tensor<32x256xf32, #mma>) -> !mtgpu.sqmma_accumulator<tensor<32x256xf32, #mma>> {
    %0 = mtgpu.pack_sqmma_accumulator %arg0 : tensor<32x256xf32, #mma> -> !mtgpu.sqmma_accumulator<tensor<32x256xf32, #mma>>
    tt.return %0 : !mtgpu.sqmma_accumulator<tensor<32x256xf32, #mma>>
  }
}
