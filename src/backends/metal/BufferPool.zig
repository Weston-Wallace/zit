const std = @import("std");
const Allocator = std.mem.Allocator;
const metal = @import("metal");
const zit = @import("../../zit.zig");
const TensorError = zit.TensorError;
const utils = @import("../utils.zig");

/// Metal buffer pool for efficient buffer reuse
const BufferPool = @This();

device: metal.Device,
allocator: Allocator,
buffers: std.AutoHashMap(usize, std.ArrayList(*metal.Buffer)),

pub fn init(device: metal.Device, allocator: Allocator) BufferPool {
    return .{
        .device = device,
        .allocator = allocator,
        .buffers = std.AutoHashMap(usize, std.ArrayList(*metal.Buffer)).init(allocator),
    };
}

pub fn deinit(self: *BufferPool) void {
    var it = self.buffers.valueIterator();
    while (it.next()) |buffer_list| {
        for (buffer_list.items) |buffer| {
            buffer.deinit();
            self.allocator.destroy(buffer);
        }
        buffer_list.deinit();
    }
    self.buffers.deinit();
}

/// Get a buffer of the specified size
pub fn getBuffer(self: *BufferPool, size: usize, mode: metal.Buffer.ResourceStorageMode) !*metal.Buffer {
    // Round up to the next power of 2 to reduce fragmentation and increase reuse
    const aligned_size = utils.nextPowerOf2(size);

    // Check if we have a buffer of this size available
    if (self.buffers.getPtr(aligned_size)) |buffer_list| {
        if (buffer_list.pop()) |buffer| {
            return buffer;
        }
    }

    // Create a new buffer if none available
    const buffer_ptr = try self.allocator.create(metal.Buffer);
    buffer_ptr.* = try self.device.createBuffer(aligned_size, mode);
    return buffer_ptr;
}

/// Return a buffer to the pool for reuse
pub fn returnBuffer(self: *BufferPool, buffer: *metal.Buffer) !void {
    const size = buffer.getLength();

    // Get or create the list for this size
    const gop = try self.buffers.getOrPut(size);
    if (!gop.found_existing) {
        gop.value_ptr.* = std.ArrayList(*metal.Buffer).init(self.allocator);
    }
    var buffer_list = gop.value_ptr;

    try buffer_list.append(buffer);
}
