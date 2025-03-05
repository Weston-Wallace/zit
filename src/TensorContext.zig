const std = @import("std");
const Allocator = std.mem.Allocator;
const Backend = @import("backend.zig").Backend;

backend: Backend = .cpu,
allocator: Allocator,
