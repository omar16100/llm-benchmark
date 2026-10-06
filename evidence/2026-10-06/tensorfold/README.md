# TensorFold native checks

Source: `zig-flashnext` at `88c424e5350a702372771915507bcbed9d513546`.
Environment: M3 Ultra, macOS 26.3 (25D125), Xcode 26.5 (17F42), Zig 0.17.0.

## Native weight reads

The probe calls `Checkpoint.addFile`, `Run.load`, and `Run.group`.
Each temporary fixture contains 2 MiB of synthetic uint32 values on a NAS mount.
The tests cover tensor offsets 0 and 64.
The delayed mode adds 5 ms per read and limits each read to 32 KiB.

All 18 forced-read cases reproduced a Metal buffer leak in the stock loaders.
The faults comprise premature EOF, `EIO`, and `EINTR` on each read path and offset.
Each case returns `ShortRead` but retains about 2 MiB of Metal allocation.
The [correction](weight-read-fix.patch) releases each buffer if its read fails.
Checkpoint ownership transfers to the shard list before the tensor index step.

All 54 cases pass after the correction.
The 18 fault cases leave Metal `currentAllocatedSize` unchanged.
The 36 payload cases preserve all 524,288 values through a Metal copy after source-file overwrite.
The shim records no subsequent `pread` call.
[Before](weight-reads-before.json) and [after](weight-reads-after.json) contain the source hashes and each result.

All 46 build steps pass for `tf-weight-read-check`, `test`, and `test-native`.
The final build executes 19 tests; the other host test targets use cached results.
[Build log](leak-build-green.log).

The page-cache state is uncontrolled.
These small tests do not establish cold NAS reliability or full-model startup performance.
The tests do not cover Kimi, CUDA, or another Apple chip.

1. Apply `weight-read-fix.patch` to the pinned source.
2. Build the probe with `zig build tf-weight-read-check -Dcpu=apple_m1`.
3. Execute `python3 tools/zig/check_weight_reads.py --output /tmp/tensorfold-weight-results`.

Use `--storage-directory /path/to/storage` to select another filesystem.
The driver removes its synthetic files and retains its logs in the new output directory.

## Native model checks

The capabilities list contains Nemotron 4-bit and FlashNext 6-bit.
The source has no native GLM, Ling, or Gemma backend.
The actual Gemma CLI reports no backend.
The Ling CLI stops earlier with `UnsupportedRegex`.
The GLM CLI exits with allocator diagnostics; those logs do not establish its failure stage.
Their original numerical issues have no executable native backend in this revision.

The full Nemotron server exits with `KernelCompile` before its health check.
Three fresh runtime compiler processes reject `uint4b_format` in the same stock shader on this Mac.
The command-line Metal compiler accepts that shader with `-std=metal4.0`.
The portable probe also reproduces the runtime error.
This establishes a runtime-versus-offline compiler difference without model weights.
The precise SDK or driver cause remains unknown.
No serial-versus-drafted token comparison or decode measurement occurred.
[Runtime result](runtime-compile.json), [offline result](compile-probe.json), [startup result](native-inference.json).

## Compiler reproduction

Execute these commands from the pinned TensorFold source directory with Zig 0.17.0.
Replace `/path/to/evidence` with the directory that contains [runtime_compile_check.zig](runtime_compile_check.zig).

```sh
zig build-exe -O ReleaseFast -mcpu=apple_m1 --dep metal -Mroot=/path/to/evidence/runtime_compile_check.zig -lc -lobjc -framework Metal -framework Foundation -Mmetal=zig/src/metal/metal.zig -femit-bin=runtime-compile-check
./runtime-compile-check zig/kernels/metal/nemotron/coop_eh_1_0_sk.metal
xcrun -sdk macosx metal -std=metal4.0 -fmetal-math-mode=safe -fmetal-math-fp32-functions=fast -c zig/kernels/metal/nemotron/coop_eh_1_0_sk.metal -o compile-probe.air
```

The probe uses the native engine's `CompileOptions.mlx()` and its `Library.fromSource` function.
[Runtime log](runtime-portable.log) and [offline log](compile-probe.log) record the compiler results.

## Legacy Python evidence

The [legacy patch](legacy-loader.patch) evaluates stored arrays before GPU transforms and preserves lazy expert placeholders.
Its selected suite passed 53 tests; one real-checkpoint test skipped.
It does not establish a NAS timeout repair.
Upstream freezes Python changes at 0.6.5 and accepts new PRs against `zig-flashnext`.

The TensorFold patches retain the upstream [Apache-2.0 license](LICENSE) and [notice](NOTICE).

The native correction is [PR 450](https://github.com/ashhart/TensorFold/pull/450).
The legacy Python correction remains available as a [fork commit](https://github.com/omar16100/TensorFold/commit/df66e87aabfcaedc2cd76c2e94badefead37c9e5).
