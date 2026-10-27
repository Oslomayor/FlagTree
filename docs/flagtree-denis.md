# 智源 FLagTree 编译器

### 环境配置

FlagTree Repo

```sh
git clone --recursive https://github.com/flagos-ai/FlagTree.git  
```

注意：如果要提PR，需要先fork，然后拉取自己github的仓库。除非有读写社区仓库权限。

安装 flagtree

```sh
cd FlagTree
LLVM_SYSPATH=${your_llvm_path} FLAGTREE_BACKEND=nvidia pip install . --no-build-isolation -v
```

PR代码格式要求

```sh
pip install pre-commit

cd FlagGems && pre-commit install
```

安装完成后每次执行git commit会自动检测代码格式。

### 提交PR合入社区仓库

本地执行 flagtree CI 跑通，本地执行必要测试用例跑通。

```sh
cd third_party/xxx_backend/python/test && pytest --device [xxx] ./unit
```

### 打包工具链

打包以下两个工具包，上传智源对应服务器，flagtree 构建能拉取到。

- llvm project

打包llvm工具链，文件名为 `nvidia-llvm22.tar `，内部文件夹名`nvidia-llvm22.tar`

```sh
cd triton_xxx/third_party/llvm-project
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DLLVM_ENABLE_TERMINFO=OFF -DCMAKE_INSTALL_PREFIX=`pwd`/llvm -DLLVM_INSTALL_UTILS=ON && ninja install
mv llvm nvidia-llvm22
tar -cvf nvidia-llvm22.tar llvm
```

- local binary

从后端编译器build目录打包 llc (disable asm)，ld.lld （lld 的 GNU/ELF 风味，兼容 GNU 的 ld 链接器）

文件名为`nvidia_local_binary.tar`，内部文件夹名`nvidia_local_binary`

文件夹内含有 llc , ld.lld 工具。
