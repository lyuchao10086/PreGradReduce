# PReluGradReduce Iteration 7 Acceptance Final

## Local Iteration i03

This iteration builds on `p_relu_grad_reduce_i02` and optimizes
`REDUCE_MODE_NCHW_CHANNEL`.

Optimized path:

- Host tiling now computes `reduceSplit` for NCHW channel reduce when
  `outputLength < coreNum`.
- Kernel maps active blocks to `(channel, reduceSlice)` instead of only output
  chunks.
- Each block reduces a slice of the `N*H*W` reduce dimension for one channel.
  The kernel coalesces each slice into row-local contiguous spans and uses the
  i02 shared-reduce `SumRange*` helpers.
- Partial sums are stored in FP32 workspace at
  `partial[channel * reduceSplit + reduceSlice]`.
- `SyncAll` is used only when `reduceSplit > 1`; one reducer block per channel
  merges partial sums and writes final `da[channel]`.

Parallel split:

- `blockDim = outputLength * reduceSplit`, capped by available Vector cores.
- `reduceSplit = blockDim / outputLength`.
- `blockDim=1` or `reduceSplit=1` falls back to the existing output-parallel
  path.

Workspace layout for this mode:

- `[sysWorkspaceSize, + blockDim * 32B)` for `SyncAll`.
- Next 32B-aligned range for `outputLength * reduceSplit` FP32 partial sums.

Unchanged paths:

- `REDUCE_MODE_ALL` keeps the i02 vectorized local sum.
- `REDUCE_MODE_ND_WEIGHT_SHAPE`
- `REDUCE_MODE_NC1HWC0`

Scalar fallback remains for unaligned heads/tails inside `SumRange*`, final
partial merge, final output write, and BF16 byte-safe UB-to-FP32 conversion.
Priority regression cases are `nchw_channel_large_fp32`,
`nchw_channel_large_fp16`, and `nchw_channel_large_bf16`.

## Local Iteration i02

This iteration builds on `p_relu_grad_reduce_i01` and optimizes only
`REDUCE_MODE_ALL`.

Optimized path:

- Shared/all-reduce still uses the existing multi-core map-reduce strategy.
- Each active AI Core owns a contiguous slice of `updates`.
- `blockDim=1` still bypasses workspace and `SyncAll`.
- FP32 local sums now use block `DataCopy` into UB plus vector `Add` folding.
- FP16 local sums now use block `DataCopy`, vector `Cast` to FP32, then vector
  `Add` folding.
- BF16 local sums now use block `DataCopy` into UB and then keep the existing
  byte-safe BF16-to-FP32 conversion before vector `Add` folding. This preserves
  the current strict-aliasing and rounding safety posture while removing the
  dominant GM scalar read path.

Unchanged paths:

- `REDUCE_MODE_NCHW_CHANNEL`
- `REDUCE_MODE_ND_WEIGHT_SHAPE`
- `REDUCE_MODE_NC1HWC0`

Scalar fallback remains for unaligned heads and tails, plus BF16 UB-to-FP32
conversion. The main aligned shared-reduce path is no longer dominated by
per-element GM reads.

## Local Iteration i01

This workspace now starts a new local optimization series using the directory and
Git name `p_relu_grad_reduce_i01`.

Iteration i01 is a baseline harness iteration. It does not change kernel
semantics or performance code. It restores large-shape correctness and benchmark
coverage in `examples/test_aclnn_p_relu_grad_reduce.cpp` so later optimization
iterations can compare `avg_us` against the same cases.

Covered benchmark groups:

- Shared all-reduce: `updates=[32,10,32,128]`, `weights=[1]`.
- NCHW channel reduce: `updates=[32,10,32,128]`, `weights=[10]`.
- ND rank3 weight-shape: `updates=[20,32,125]`, `weights=[32,125]`.
- ND rank4 weight-shape: `updates=[32,10,32,128]`, `weights=[10,32,128]`.
- NC1HWC0: `updates=[32,2,32,128,16]`, `weights=[2,1,1,16]`.
- FP32, FP16, and BF16 variants are included for each group.

Rank4 large weight-shape cases remain gated in default runs. Set
`PRELU_GRAD_REDUCE_RUN_SLOW=1` for full acceptance. Set
`PRELU_GRAD_REDUCE_BENCH=1` and optionally `PRELU_GRAD_REDUCE_CASE=<case>` to
collect per-case timing.

The old root `test_prelu_grad_reduce.json` metadata file was intentionally
removed by the project owner before i01 and is not restored in this iteration.

This project is the seventh and final iteration copy, based on `p_relu_grad_reduce_i06_multicore_perf`.

Iteration 7 keeps all Iteration 6 functionality, fixes the final shared-reduce workspace edge case, and closes the acceptance checklist:

- Shared/all-reduce cases no longer force `blockDim=1` when the input is large.
- Each AI Core computes a partial sum over a contiguous slice of `updates`.
- Partial sums are stored in workspace, synchronized, and reduced by block 0 before writing `da[0]`.
- `blockDim=1` shared/all-reduce bypasses the workspace partial-sum path and writes output directly.
- Existing output-parallel channel, ND weight-shape, and NC1HWC0 paths remain unchanged.
- The example runner still reports both absolute and relative error.
- The example runner prints a final `[ACCEPTANCE]` summary line after all cases pass.
- Full TBE performance parity still requires cloud profiling evidence to fill the report fields.
- It records the current large-shape acceptance cases directly in `examples/test_aclnn_p_relu_grad_reduce.cpp`.
- The matching TBE-side case metadata is kept in `../prelu_grad_reduce.py`.
- The original `p_relu_grad_reduce` directory must remain unchanged.

## Iteration 7 Status

Current large-shape baseline coverage:

- NCHW large FP32/FP16/BF16: `updates=[32,10,32,128]`, `weights=[10]`.
- NCHW signed FP32: `updates=[8,10,17,19]`, `weights=[10]`.
- NC1HWC0 large FP32/FP16/BF16: `updates=[32,2,32,128,16]`, `weights=[2,1,1,16]`.
- NC1HWC0 signed BF16: `updates=[4,2,17,19,16]`, `weights=[2,1,1,16]`.
- ND rank3 large FP32/FP16/BF16: `updates=[20,32,125]`, `weights=[32,125]`.
- ND rank3 cancellation FP16: `updates=[18,32,33]`, `weights=[32,33]`.
- ND rank2 channel FP32/FP16/BF16: `updates=[64,32]`, `weights=[32]`.
- ND rank3 channel FP32/FP16/BF16: `updates=[20,32,125]`, `weights=[32]`.
- Shared large FP32/FP16/BF16: `updates=[32,10,32,128]`, `weights=[1]`.
- ND rank4 large FP32/FP16/BF16: `updates=[32,10,32,128]`, `weights=[10,32,128]`.
- Shared rank3 FP32/FP16/BF16: `updates=[20,32,125]`, `weights=[1]`.
- FP16/BF16 cases use `inputScale` where needed to avoid output overflow while preserving large-shape pressure.

Cloud-side evidence still needed for formal acceptance:

- Build i07 in CANNLab.
- Run all Ascend C cases and collect `[REPORT]` lines.
- Build/run the matching TBE cases from `../prelu_grad_reduce.py`.
- Fill `TBE截图`, `Ascend截图`, `TBE算子性能(us)`, and `Ascend C算子性能(us)` from profiling output.

The self-verification report fields follow the uploaded `FloorMod算子自验证报告.xlsx` header:

```text
shape, dtype, 精度是否通过, TBE截图, Ascend截图, TBE算子性能(us), Ascend C算子性能(us), 解释说明
```

The C++ runner prints a `[REPORT]` line with these fields after each case.

When this iteration is copied into an ops-nn workspace, keep the directory name
`p_relu_grad_reduce_i07_acceptance_final` and build it as:

```bash
bash build.sh --pkg --soc=ascend910b --ops=p_relu_grad_reduce_i07_acceptance_final -j16
bash build.sh --run_example p_relu_grad_reduce_i07_acceptance_final eager cust --vendor_name=custom
```

---

# Original PReluGradReduce ops-nn Migration Notes

This directory is an ops-nn style migration of the standalone `PReluGradReduce`
kernel that has already passed CPU and NPU direct-invocation tests.

## Scope

The first migration keeps the verified baseline narrow:

- op type: `PReluGradReduce`
- ops-nn directory name: `p_relu_grad_reduce`
- input tensors: `grads`, `features`, `weights`, `updates`
- output tensor: `da`
- supported runtime case: `float16`, `ND`, `weights` shape `[1]`
- compute: `da[0] = reduce_sum(updates)`
- tiling: single AI Core, `totalNum` read from `updates` shape

`grads`, `features`, and `weights` are preserved for interface compatibility.
The first kernel implementation only reads `updates` and writes `da`.

## Files

```text
p_relu_grad_reduce
├── CMakeLists.txt
├── examples
│   └── test_aclnn_p_relu_grad_reduce.cpp
├── op_graph
│   ├── CMakeLists.txt
│   ├── fusion_pass
│   └── p_relu_grad_reduce_proto.h
├── op_host
│   ├── CMakeLists.txt
│   ├── p_relu_grad_reduce_def.cpp
│   ├── p_relu_grad_reduce_infershape.cpp
│   └── p_relu_grad_reduce_tiling.cpp
└── op_kernel
    ├── p_relu_grad_reduce.cpp
    ├── p_relu_grad_reduce.h
    ├── p_relu_grad_reduce_tiling_data.h
    └── p_relu_grad_reduce_tiling_key.h
```

## Install Into ops-nn

Copy this directory into `ops-nn/examples` and make sure the destination folder
is named `p_relu_grad_reduce`:

```bash
cd /mnt/workspace/gitCode/${gitCode_id}/ops-nn
cp -r /path/to/p_relu_grad_reduce ./examples/p_relu_grad_reduce
```

From this workspace, copy it directly:

```bash
cp -r /Users/lyuchao/Desktop/codex/prelu_grad_reduce/p_relu_grad_reduce ./examples/p_relu_grad_reduce
```

## Build

Atlas A3 / 910_93:

```bash
bash build.sh --pkg --soc=ascend910_93 --ops=p_relu_grad_reduce -j16
```

Atlas A2 / 910B:

```bash
bash build.sh --pkg --soc=ascend910b --ops=p_relu_grad_reduce -j16
```

## Install

```bash
./build_out/cann-ops-nn-*linux*.run
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_nn/op_api/lib:${LD_LIBRARY_PATH}
```

## Run Example

```bash
bash build.sh --run_example p_relu_grad_reduce eager cust --vendor_name=custom
```

Expected success message:

```text
p_relu_grad_reduce expected: <value>, actual: <value>, diff: <value>
test pass
```

## Next Steps

After this first package compiles and the example passes, extend in this order:

1. Support `float32` and `bfloat16`.
2. Replace the single-core loop with multi-core partial reduce plus second-stage reduce.
3. Generalize `weights` shapes and reduce axes using the old TBE implementation as the shape-rule reference.
4. Add formal unit tests under `tests/ut`.
