const std = @import("std");
const Allocator = std.mem.Allocator;
const metal = @import("metal");
const zit = @import("../../zit.zig");
const TensorError = zit.TensorError;
const shaders = @import("shaders.zig");

// Global context for the Metal backend
var context: ?MetalContext = null;

// Metal context holds all the device, command queue, and compiled shaders
pub const MetalContext = struct {
    allocator: Allocator,
    device: metal.Device,
    command_queue: metal.CommandQueue,

    // Shader libraries
    elementwise_lib: metal.Library,
    vector_ops_lib: metal.Library,
    matrix_vector_lib: metal.Library,
    matrix_lib: metal.Library,

    // Pipeline states for operations
    add_pipeline: metal.ComputePipelineState,
    subtract_pipeline: metal.ComputePipelineState,
    multiply_pipeline: metal.ComputePipelineState,
    divide_pipeline: metal.ComputePipelineState,
    scalar_multiply_pipeline: metal.ComputePipelineState,
    vector_dot_pipeline: metal.ComputePipelineState,
    vector_norm_pipeline: metal.ComputePipelineState,
    matrix_vector_multiply_pipeline: metal.ComputePipelineState,
    matrix_multiply_pipeline: metal.ComputePipelineState,
    matrix_transpose_pipeline: metal.ComputePipelineState,

    fn init(allocator: Allocator) !MetalContext {
        var device = try metal.Device.createDefault();
        errdefer device.deinit();

        var command_queue = try device.createCommandQueue();
        errdefer command_queue.deinit();

        var ctx = MetalContext{
            .allocator = allocator,
            .device = device,
            .command_queue = command_queue,
            // These will be initialized in setupPipelines
            .elementwise_lib = undefined,
            .vector_ops_lib = undefined,
            .matrix_vector_lib = undefined,
            .matrix_lib = undefined,
            .add_pipeline = undefined,
            .subtract_pipeline = undefined,
            .multiply_pipeline = undefined,
            .divide_pipeline = undefined,
            .scalar_multiply_pipeline = undefined,
            .vector_dot_pipeline = undefined,
            .vector_norm_pipeline = undefined,
            .matrix_vector_multiply_pipeline = undefined,
            .matrix_multiply_pipeline = undefined,
            .matrix_transpose_pipeline = undefined,
        };

        try ctx.setupPipelines();

        return ctx;
    }

    fn setupPipelines(self: *MetalContext) !void {
        // Compile elementwise operations shader
        const elementwise_result = try self.device.createLibraryFromSource(shaders.elementwise_shader, self.allocator);
        if (elementwise_result.error_msg) |err_msg| {
            defer self.allocator.free(err_msg);
            std.log.err("Failed to compile elementwise shader: {s}", .{err_msg});
            return TensorError.BackendError;
        }
        self.elementwise_lib = elementwise_result.library;

        // Create elementwise pipeline states
        self.add_pipeline = try self.createPipelineState("add", &self.elementwise_lib);
        self.subtract_pipeline = try self.createPipelineState("subtract", &self.elementwise_lib);
        self.multiply_pipeline = try self.createPipelineState("multiply", &self.elementwise_lib);
        self.divide_pipeline = try self.createPipelineState("divide", &self.elementwise_lib);
        self.scalar_multiply_pipeline = try self.createPipelineState("scalar_multiply", &self.elementwise_lib);

        // Compile vector operations shader
        const vector_result = try self.device.createLibraryFromSource(shaders.vector_ops_shader, self.allocator);
        if (vector_result.error_msg) |err_msg| {
            defer self.allocator.free(err_msg);
            std.log.err("Failed to compile vector ops shader: {s}", .{err_msg});
            return TensorError.BackendError;
        }
        self.vector_ops_lib = vector_result.library;

        // Create vector pipeline states
        self.vector_dot_pipeline = try self.createPipelineState("vector_dot", &self.vector_ops_lib);
        self.vector_norm_pipeline = try self.createPipelineState("vector_norm", &self.vector_ops_lib);

        // Compile matrix-vector operations shader
        const matrix_vector_result = try self.device.createLibraryFromSource(shaders.matrix_vector_ops_shader, self.allocator);
        if (matrix_vector_result.error_msg) |err_msg| {
            defer self.allocator.free(err_msg);
            std.log.err("Failed to compile matrix-vector ops shader: {s}", .{err_msg});
            return TensorError.BackendError;
        }
        self.matrix_vector_lib = matrix_vector_result.library;

        // Create matrix-vector pipeline states
        self.matrix_vector_multiply_pipeline = try self.createPipelineState("matrix_vector_multiply", &self.matrix_vector_lib);

        // Compile matrix operations shader
        const matrix_result = try self.device.createLibraryFromSource(shaders.matrix_ops_shader, self.allocator);
        if (matrix_result.error_msg) |err_msg| {
            defer self.allocator.free(err_msg);
            std.log.err("Failed to compile matrix ops shader: {s}", .{err_msg});
            return TensorError.BackendError;
        }
        self.matrix_lib = matrix_result.library;

        // Create matrix pipeline states
        self.matrix_multiply_pipeline = try self.createPipelineState("matrix_multiply", &self.matrix_lib);
        self.matrix_transpose_pipeline = try self.createPipelineState("matrix_transpose", &self.matrix_lib);
    }

    fn createPipelineState(self: *MetalContext, function_name: []const u8, library: *metal.Library) !metal.ComputePipelineState {
        var function = try library.getFunction(function_name, self.allocator);
        defer function.deinit();

        return try function.createComputePipelineState();
    }

    fn deinit(self: *MetalContext) void {
        // Release all pipelines
        self.add_pipeline.deinit();
        self.subtract_pipeline.deinit();
        self.multiply_pipeline.deinit();
        self.divide_pipeline.deinit();
        self.scalar_multiply_pipeline.deinit();
        self.vector_dot_pipeline.deinit();
        self.vector_norm_pipeline.deinit();
        self.matrix_vector_multiply_pipeline.deinit();
        self.matrix_multiply_pipeline.deinit();
        self.matrix_transpose_pipeline.deinit();

        // Release libraries
        self.elementwise_lib.deinit();
        self.vector_ops_lib.deinit();
        self.matrix_vector_lib.deinit();
        self.matrix_lib.deinit();

        // Release command queue and device
        self.command_queue.deinit();
        self.device.deinit();
    }
};

// Initialize the Metal context
pub fn init(allocator: Allocator) !void {
    if (context != null) return;

    context = try MetalContext.init(allocator);
}

// Get the Metal context
pub fn get() ?*MetalContext {
    if (context) |*ctx| {
        return ctx;
    }
    return null;
}

// Check if Metal is available on this system
pub fn isAvailable() bool {
    if (context == null) {
        // Try to create a Metal device just to check availability
        const device = metal.Device.createDefault() catch {
            return false;
        };
        device.deinit();
        return true;
    }
    return true;
}

// Clean up Metal resources
pub fn deinit() void {
    if (context) |*ctx| {
        ctx.deinit();
        context = null;
    }
}

test {
    const device = metal.Device.createDefault();
    const name = try device.getCName();
    defer metal.freeCString(name);
    std.debug.print("name: {s}\n", .{name});
}
