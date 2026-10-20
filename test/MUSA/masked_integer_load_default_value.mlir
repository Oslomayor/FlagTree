// RUN: triton-opt %s --convert-triton-musagpu-to-llvm=compute-capability=31 -reconcile-unrealized-casts | FileCheck %s
// RUN: triton-opt %s --convert-triton-musagpu-to-llvm=compute-capability=32 -reconcile-unrealized-casts | FileCheck %s --check-prefix=PH1S

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {
  "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32,
  ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32
} {
  tt.func public @masked_integer_default(%src: !tt.ptr<i32>, %mask: i1) {
    %ptrs = tt.splat %src : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %m = tt.splat %mask : i1 -> tensor<1xi1, #blocked>
    %v = tt.load %ptrs, %m : tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %ptrs, %v, %m : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  tt.func public @masked_integer_divisor(%src: !tt.ptr<i32>, %mask: i1) {
    %c2 = arith.constant 2 : i32
    %ptrs = tt.splat %src : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %m = tt.splat %mask : i1 -> tensor<1xi1, #blocked>
    %num = tt.splat %c2 : i32 -> tensor<1xi32, #blocked>
    %den = tt.load %ptrs, %m : tensor<1x!tt.ptr<i32>, #blocked>
    %rem = arith.remsi %num, %den : tensor<1xi32, #blocked>
    tt.store %ptrs, %rem, %m : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  tt.func public @masked_integer_divisor_through_add(%src: !tt.ptr<i32>, %mask: i1) {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %ptrs = tt.splat %src : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %m = tt.splat %mask : i1 -> tensor<1xi1, #blocked>
    %zero = tt.splat %c0 : i32 -> tensor<1xi32, #blocked>
    %num = tt.splat %c2 : i32 -> tensor<1xi32, #blocked>
    %den0 = tt.load %ptrs, %m : tensor<1x!tt.ptr<i32>, #blocked>
    %den = arith.addi %den0, %zero : tensor<1xi32, #blocked>
    %rem = arith.remsi %num, %den : tensor<1xi32, #blocked>
    tt.store %ptrs, %rem, %m : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  tt.func public @masked_integer_vector_divisor(
      %src: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked4>
    %base = tt.splat %src : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked4>
    %ptrs = tt.addptr %base, %range : tensor<4x!tt.ptr<i32>, #blocked4>, tensor<4xi32, #blocked4>
    %true = arith.constant true
    %m = tt.splat %true : i1 -> tensor<4xi1, #blocked4>
    %c1 = arith.constant 1 : i32
    %num = tt.splat %c1 : i32 -> tensor<4xi32, #blocked4>
    %den = tt.load %ptrs, %m : tensor<4x!tt.ptr<i32>, #blocked4>
    %rem = arith.remsi %num, %den : tensor<4xi32, #blocked4>
    tt.store %ptrs, %rem, %m : tensor<4x!tt.ptr<i32>, #blocked4>
    tt.return
  }

  tt.func public @masked_integer_explicit(%src: !tt.ptr<i32>, %mask: i1) {
    %c0 = arith.constant 0 : i32
    %ptrs = tt.splat %src : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %m = tt.splat %mask : i1 -> tensor<1xi1, #blocked>
    %other = tt.splat %c0 : i32 -> tensor<1xi32, #blocked>
    %v = tt.load %ptrs, %m, %other : tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %ptrs, %v, %m : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  tt.func public @masked_float_default(%src: !tt.ptr<f32>, %mask: i1) {
    %ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %m = tt.splat %mask : i1 -> tensor<1xi1, #blocked>
    %v = tt.load %ptrs, %m : tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %v, %m : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @masked_bf16_default(%src: !tt.ptr<bf16>, %mask: i1) {
    %ptrs = tt.splat %src : !tt.ptr<bf16> -> tensor<1x!tt.ptr<bf16>, #blocked>
    %m = tt.splat %mask : i1 -> tensor<1xi1, #blocked>
    %v = tt.load %ptrs, %m : tensor<1x!tt.ptr<bf16>, #blocked>
    tt.store %ptrs, %v, %m : tensor<1x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @masked_bf16_explicit(%src: !tt.ptr<bf16>, %mask: i1) {
    %one = arith.constant 1.000000e+00 : bf16
    %ptrs = tt.splat %src : !tt.ptr<bf16> -> tensor<1x!tt.ptr<bf16>, #blocked>
    %m = tt.splat %mask : i1 -> tensor<1xi1, #blocked>
    %other = tt.splat %one : bf16 -> tensor<1xbf16, #blocked>
    %v = tt.load %ptrs, %m, %other : tensor<1x!tt.ptr<bf16>, #blocked>
    tt.store %ptrs, %v, %m : tensor<1x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @masked_bf16_true(%src: !tt.ptr<bf16>) {
    %true = arith.constant true
    %ptrs = tt.splat %src : !tt.ptr<bf16> -> tensor<1x!tt.ptr<bf16>, #blocked>
    %m = tt.splat %true : i1 -> tensor<1xi1, #blocked>
    %v = tt.load %ptrs, %m : tensor<1x!tt.ptr<bf16>, #blocked>
    tt.store %ptrs, %v, %m : tensor<1x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @masked_bf16_false(%src: !tt.ptr<bf16>) {
    %false = arith.constant false
    %ptrs = tt.splat %src : !tt.ptr<bf16> -> tensor<1x!tt.ptr<bf16>, #blocked>
    %m = tt.splat %false : i1 -> tensor<1xi1, #blocked>
    %v = tt.load %ptrs, %m : tensor<1x!tt.ptr<bf16>, #blocked>
    tt.store %ptrs, %v, %m : tensor<1x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @masked_fp8_default(%src: !tt.ptr<f8E4M3FN>, %mask: i1) {
    %ptrs = tt.splat %src : !tt.ptr<f8E4M3FN> -> tensor<1x!tt.ptr<f8E4M3FN>, #blocked>
    %m = tt.splat %mask : i1 -> tensor<1xi1, #blocked>
    %v = tt.load %ptrs, %m : tensor<1x!tt.ptr<f8E4M3FN>, #blocked>
    tt.store %ptrs, %v, %m : tensor<1x!tt.ptr<f8E4M3FN>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: llvm.func @masked_integer_default
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(dense<0> : vector<1xi32>)
// CHECK: llvm.cond_br {{.*}}, ^bb1, ^bb2(%[[ZERO]] : vector<1xi32>)
// CHECK: llvm.load {{.*}} : !llvm.ptr<1> -> vector<1xi32>
// CHECK: llvm.extractelement
// CHECK-LABEL: llvm.func @masked_integer_divisor
// CHECK: %[[ZERO_DIV:.*]] = llvm.mlir.constant(dense<0> : vector<1xi32>)
// CHECK: llvm.cond_br {{.*}}, ^bb1, ^bb2(%[[ZERO_DIV]] : vector<1xi32>)
// CHECK: llvm.extractelement
// CHECK: llvm.srem
// CHECK-LABEL: llvm.func @masked_integer_divisor_through_add
// CHECK: %[[ZERO_ADD:.*]] = llvm.mlir.constant(dense<0> : vector<1xi32>)
// CHECK: llvm.cond_br {{.*}}, ^bb1, ^bb2(%[[ZERO_ADD]] : vector<1xi32>)
// CHECK: llvm.extractelement
// CHECK: llvm.srem
// CHECK-LABEL: llvm.func @masked_integer_vector_divisor
// CHECK: llvm.mlir.constant(dense<0> : vector<4xi32>)
// CHECK-LABEL: llvm.func @masked_integer_explicit
// CHECK: llvm.mlir.constant(0 : i32)
// CHECK-NOT: llvm.mlir.constant(1 : i32)
// CHECK-LABEL: llvm.func @masked_float_default
// CHECK: llvm.mlir.constant(0.000000e+00 : f32)
// CHECK-LABEL: llvm.func @masked_bf16_default
// CHECK: %[[BF16_ZERO:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<1xbf16>)
// CHECK: llvm.cond_br {{.*}}, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}(%[[BF16_ZERO]] : vector<1xbf16>)
// CHECK: llvm.load {{.*}} : !llvm.ptr<1> -> vector<1xbf16>
// CHECK: llvm.extractelement
// CHECK-LABEL: llvm.func @masked_bf16_explicit
// CHECK: llvm.cond_br {{.*}} {{.*}} : vector<1xbf16>
// CHECK: llvm.load {{.*}} : !llvm.ptr<1> -> vector<1xbf16>
// CHECK: llvm.extractelement
// CHECK-LABEL: llvm.func @masked_bf16_true
// CHECK: llvm.load {{.*}} : !llvm.ptr<1> -> bf16
// CHECK-NOT: vector<1xbf16>
// CHECK-LABEL: llvm.func @masked_bf16_false
// CHECK-NOT: vector<1xbf16>
// CHECK-LABEL: llvm.func @masked_fp8_default
// CHECK: llvm.mlir.constant(0 : i8)

// PH1S-LABEL: llvm.func @masked_bf16_default
// PH1S: llvm.mlir.constant(0.000000e+00 : bf16)
// PH1S: llvm.cond_br {{.*}} {{.*}} : bf16
// PH1S: llvm.load {{.*}} : !llvm.ptr<1> -> bf16
// PH1S-NOT: vector<1xbf16>
