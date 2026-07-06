// RUN: triton-opt %s --convert-triton-musagpu-to-llvm=compute-capability=31 -reconcile-unrealized-casts | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func {{.*}}@f32_to_f16_trunc
  // CHECK-NOT: llvm.call_intrinsic "llvm.musa.f2h.rn"
  // CHECK: llvm.fptrunc
  tt.func private @f32_to_f16_trunc(%arg0: tensor<1xf32, #blocked>) -> tensor<1xf16, #blocked> {
    %0 = arith.truncf %arg0 : tensor<1xf32, #blocked> to tensor<1xf16, #blocked>
    tt.return %0 : tensor<1xf16, #blocked>
  }

  // CHECK-LABEL: llvm.func {{.*}}@f32_to_f16_rtne_fp_to_fp
  // CHECK: llvm.call_intrinsic "llvm.musa.f2h.rn"
  // CHECK-NOT: llvm.fptrunc
  tt.func private @f32_to_f16_rtne_fp_to_fp(%arg0: tensor<1xf32, #blocked>) -> tensor<1xf16, #blocked> {
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<1xf32, #blocked> -> tensor<1xf16, #blocked>
    tt.return %0 : tensor<1xf16, #blocked>
  }
}
