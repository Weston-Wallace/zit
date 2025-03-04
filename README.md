# ZIT: Zig Tensor Library

A high-performance numerical computing library for the Zig programming language, focused on tensor operations and linear algebra.

## Goals

- Provide a clean, efficient tensor library for scientific computing in Zig
- Support multiple backends (CPU SIMD, GPU via Metal/WebGPU)
- Deliver best-in-class performance for numerical operations
- Serve as a foundation for machine learning and data processing applications
- Maintain Zig-native design principles: explicit, reusable, comprehensible

## Implementation Plan

### Phase 1: Core Tensor Implementation

- [x] Set up basic project structure
- [x] Define initial Tensor type
- [ ] Implement tensor creation functions
  - [ ] Zeros, ones, eye, random initialization
  - [ ] From slice/array
  - [ ] Implement proper memory management with allocator pattern
- [ ] Add shape manipulation capabilities
  - [ ] Reshape, transpose, concat, stack
  - [ ] Slicing and indexing
- [ ] Build comprehensive test suite
- [ ] Implement basic data loading/saving

**Expected outcome**: A usable basic tensor library with proper memory management.

### Phase 2: Element-wise Operations

- [ ] Basic element-wise operations
  - [ ] add, subtract, multiply, divide
  - [ ] pow, exp, log, sqrt, etc.
  - [ ] Function mapping and transformation
- [ ] Broadcasting implementation
  - [ ] Auto-expandable dimensions
  - [ ] Efficient memory handling during broadcasts
- [ ] Optimized vectorized versions of operations
  - [ ] SIMD acceleration
  - [ ] Parallel implementations for large tensors
- [ ] Add benchmarking suite

**Expected outcome**: A library capable of performing efficient element-wise operations with SIMD acceleration.

### Phase 3: Linear Algebra

- [ ] Matrix multiplication
  - [ ] Various multiplication algorithms
  - [ ] Cache-friendly implementations
- [ ] Decompositions
  - [ ] LU, QR, SVD, Eigendecomposition
- [ ] System solving
  - [ ] Linear systems
  - [ ] Least squares
- [ ] Additional operations
  - [ ] Determinant, trace, norm
  - [ ] Inverse, pseudo-inverse

**Expected outcome**: A comprehensive linear algebra library capable of handling most scientific computing needs.

### Phase 4: Multi-threading and Optimization

- [ ] Thread pool implementation
- [ ] Parallel algorithms for large tensor operations
- [ ] Block-based processing for cache optimization
- [ ] Memory usage optimization
  - [ ] Buffer manager for memory reuse
  - [ ] In-place operations
- [ ] Performance profiling and optimization

**Expected outcome**: Highly performant implementations that scale with available hardware.

### Phase 5: GPU Acceleration

- [ ] Metal backend (macOS/iOS)
  - [ ] Kernel implementations for common operations
  - [ ] Memory management between CPU/GPU
- [ ] WebGPU backend (cross-platform)
- [ ] Automatic backend selection
- [ ] Benchmark comparison between CPU and GPU implementations

**Expected outcome**: GPU acceleration for tensor operations with significant performance improvements for large workloads.

### Phase 6: Advanced Features

- [ ] Automatic differentiation
- [ ] Statistical functions
- [ ] Signal processing capabilities
- [ ] Integration with visualization tools
- [ ] Special function support

**Expected outcome**: A feature-rich library suitable for advanced scientific computing applications.

## Interface Design

### Core Tensor Interface

```zig
pub const Tensor = struct {
    // Core properties
    data: []T,           // Underlying data storage
    shape: []usize,      // Dimensions of the tensor
    strides: []usize,    // Byte offsets for each dimension
    allocator: Allocator, // Memory allocator
    
    // Creation functions
    pub fn init(allocator: Allocator, shape: []const usize) !Tensor {...}
    pub fn zeros(allocator: Allocator, shape: []const usize) !Tensor {...}
    pub fn ones(allocator: Allocator, shape: []const usize) !Tensor {...}
    pub fn eye(allocator: Allocator, n: usize) !Tensor {...}
    pub fn fromSlice(allocator: Allocator, data: []const T, shape: []const usize) !Tensor {...}
    pub fn random(allocator: Allocator, shape: []const usize, min: T, max: T) !Tensor {...}
    
    // Memory management
    pub fn deinit(self: *Tensor) void {...}
    pub fn clone(self: Tensor) !Tensor {...}
    
    // Element access
    pub fn get(self: Tensor, indices: []const usize) !T {...}
    pub fn set(self: *Tensor, indices: []const usize, value: T) !void {...}
    
    // Shape manipulation
    pub fn reshape(self: Tensor, new_shape: []const usize) !Tensor {...}
    pub fn transpose(self: Tensor) !Tensor {...}
    pub fn slice(self: Tensor, start: []const usize, end: []const usize) !Tensor {...}
    
    // String representation for debugging
    pub fn format(self: Tensor, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {...}
};
```

### Namespaced Modules

- `zit.ops`: Element-wise operations (add, multiply, map, etc.)
- `zit.linalg`: Linear algebra operations (matmul, decompositions, etc.)
- `zit.random`: Random tensor generation
- `zit.io`: Input/output operations
- `zit.backend`: Backend management (CPU/GPU switching)

## Usage Examples

### Basic Operations

```zig
const std = @import("std");
const zit = @import("zit");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create tensors
    var a = try zit.Tensor(f32).init(allocator, &[_]usize{2, 3});
    defer a.deinit();
    
    var b = try zit.Tensor(f32).ones(allocator, &[_]usize{2, 3});
    defer b.deinit();
    
    try a.fill(5.0);
    
    // Add tensors
    var c = try zit.ops.add(a, b);
    defer c.deinit();
    
    // Matrix multiplication
    var d = try zit.Tensor(f32).fromSlice(allocator, 
        &[_]f32{1, 2, 3, 4, 5, 6}, &[_]usize{2, 3});
    defer d.deinit();
    
    var e = try zit.Tensor(f32).fromSlice(allocator,
        &[_]f32{7, 8, 9, 10, 11, 12}, &[_]usize{3, 2});
    defer e.deinit();
    
    var f = try zit.linalg.matmul(d, e);
    defer f.deinit();
    
    try zit.io.print(f);
}
```

### Memory Optimization with Buffer Manager

```zig
const std = @import("std");
const zit = @import("zit");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a buffer manager
    var manager = try zit.memory.BufferManager.init(allocator);
    defer manager.deinit();
    
    // Get tensors from the buffer manager
    var a = try manager.getOrCreate("a", &[_]usize{100, 100});
    var b = try manager.getOrCreate("b", &[_]usize{100, 100});
    var c = try manager.getOrCreate("result", &[_]usize{100, 100});
    
    // Fill with data
    try zit.random.fillUniform(a, 0, 1);
    try zit.random.fillUniform(b, 0, 1);
    
    // Use in-place operations to avoid new allocations
    try zit.linalg.matmulInPlace(a, b, c);
    try zit.ops.mapInPlace(c, sigmoid, c);
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}
```

## Performance Considerations

- **Memory layout**: Use contiguous layouts for better cache performance
- **SIMD vectorization**: Leverage Zig's SIMD capabilities for element-wise operations
- **Parallelization**: Use thread pools for large tensor operations
- **Backend switching**: Automatically select the fastest backend for each operation
- **Buffer reuse**: Minimize allocations through buffer management
- **Algorithm selection**: Different algorithms based on tensor size and shape
