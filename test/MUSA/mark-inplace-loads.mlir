// RUN: triton-opt %s -allow-unregistered-dialect --tritonmusa-mark-inplace-loads | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {
  "musa.inplace_alias_pairs" = "0:1",
  "triton_gpu.num-ctas" = 1 : i32,
  "triton_gpu.num-warps" = 4 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  triton_gpu.target = "musa:31"
} {
  // CHECK-LABEL: tt.func @runtime_alias_inplace
  tt.func @runtime_alias_inplace(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %base0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %base1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %ptr0 = tt.addptr %base0, %range : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %ptr1 = tt.addptr %base1, %range : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %limit = tt.splat %arg2 : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit : tensor<64xi32, #blocked>
    // CHECK: tt.load {{.*}} {musa.inplace_load_candidate}
    %val = tt.load %ptr0, %mask : tensor<64x!tt.ptr<f32>, #blocked>
    tt.store %ptr1, %val, %mask : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
