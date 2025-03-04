# ZIT: Zig Tensor Library

A high-performance numerical computing library for the Zig programming language, focused on tensor operations and linear algebra.

## Goals

-   Provide a clean, efficient tensor library for scientific computing in Zig
-   Support multiple backends (CPU SIMD, GPU via Metal/WebGPU)
-   Deliver best-in-class performance for numerical operations
-   Serve as a foundation for machine learning and data processing applications
-   Maintain Zig-native design principles: explicit, reusable, comprehensible

## Implementation Plan

### Phase 1: Core Tensor Implementation

-   [x] Set up basic project structure
-   [x] Define initial Tensor type
-   [ ] Implement tensor creation functions
    -   [ ] Zeros, ones, eye, random initialization
    -   [ ] From slice/array
    -   [ ] Implement proper memory management with allocator pattern
-   [ ] Add shape manipulation capabilities
    -   [ ] Reshape, transpose, concat, stack
    -   [ ] Slicing and indexing
-   [ ] Build comprehensive test suite
-   [ ] Implement basic data loading/saving

**Expected outcome**: A usable basic tensor library with proper memory management.

### Phase 2: Element-wise Operations

-   [ ] Basic element-wise operations
    -   [ ] add, subtract, multiply, divide
    -   [ ] pow, exp, log, sqrt, etc.
    -   [ ] Function mapping and transformation
-   [ ] Broadcasting implementation
    -   [ ] Auto-expandable dimensions
    -   [ ] Efficient memory handling during broadcasts
-   [ ] Optimized vectorized versions of operations
    -   [ ] SIMD acceleration
    -   [ ] Parallel implementations for large tensors
-   [ ] Add benchmarking suite

**Expected outcome**: A library capable of performing efficient element-wise operations with SIMD acceleration.

### Phase 3: Linear Algebra

-   [ ] Matrix multiplication
    -   [ ] Various multiplication algorithms
    -   [ ] Cache-friendly implementations
-   [ ] Decompositions
    -   [ ] LU, QR, SVD, Eigendecomposition
-   [ ] System solving
    -   [ ] Linear systems
    -   [ ] Least squares
-   [ ] Additional operations
    -   [ ] Determinant, trace, norm
    -   [ ] Inverse, pseudo-inverse

**Expected outcome**: A comprehensive linear algebra library capable of handling most scientific computing needs.

### Phase 4: Multi-threading and Optimization

-   [ ] Thread pool implementation
-   [ ] Parallel algorithms for large tensor operations
-   [ ] Block-based processing for cache optimization
-   [ ] Memory usage optimization
    -   [ ] Buffer manager for memory reuse
    -   [ ] In-place operations
-   [ ] Performance profiling and optimization

**Expected outcome**: Highly performant implementations that scale with available hardware.

### Phase 5: GPU Acceleration

-   [ ] Metal backend (macOS/iOS)
    -   [ ] Kernel implementations for common operations
    -   [ ] Memory management between CPU/GPU
-   [ ] WebGPU backend (cross-platform)
-   [ ] Automatic backend selection
-   [ ] Benchmark comparison between CPU and GPU implementations

**Expected outcome**: GPU acceleration for tensor operations with significant performance improvements for large workloads.

### Phase 6: Advanced Features

-   [ ] Automatic differentiation
-   [ ] Statistical functions
-   [ ] Signal processing capabilities
-   [ ] Integration with visualization tools
-   [ ] Special function support

**Expected outcome**: A feature-rich library suitable for advanced scientific computing applications.

## Performance Considerations

-   **Memory layout**: Use contiguous layouts for better cache performance
-   **SIMD vectorization**: Leverage Zig's SIMD capabilities for element-wise operations
-   **Parallelization**: Use thread pools for large tensor operations
-   **Backend switching**: Automatically select the fastest backend for each operation
-   **Buffer reuse**: Minimize allocations through buffer management
-   **Algorithm selection**: Different algorithms based on tensor size and shape
