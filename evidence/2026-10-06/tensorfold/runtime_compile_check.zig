//! Compile a source with the same runtime options as the native model.
const std = @import("std");
const mtl = @import("metal");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) return error.ExpectedSourceFile;
    const source = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1 << 20));
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const options = mtl.CompileOptions.mlx();
    std.debug.print("device={s} language={d} source_bytes={d}\n", .{ device.name(), options.language, source.len });
    const library = try mtl.Library.fromSource(device, source, options);
    defer library.deinit();
    std.debug.print("runtime_compile=pass\n", .{});
}
