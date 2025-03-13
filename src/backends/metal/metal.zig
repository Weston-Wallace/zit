const std = @import("std");
const MetalBackend = @import("MetalBackend.zig");

pub const init = MetalBackend.init;
pub const deinit = MetalBackend.deinit;
pub const isAvailable = MetalBackend.isAvailable;
pub const backend = MetalBackend.backend;
