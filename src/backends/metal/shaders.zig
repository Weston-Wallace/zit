// Element-wise operations shader
pub const elementwise_shader =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void add(
    \\    device const float* a [[buffer(0)]],
    \\    device const float* b [[buffer(1)]],
    \\    device float* result [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]])
    \\{
    \\    result[id] = a[id] + b[id];
    \\}
    \\
    \\kernel void subtract(
    \\    device const float* a [[buffer(0)]],
    \\    device const float* b [[buffer(1)]],
    \\    device float* result [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]])
    \\{
    \\    result[id] = a[id] - b[id];
    \\}
    \\
    \\kernel void multiply(
    \\    device const float* a [[buffer(0)]],
    \\    device const float* b [[buffer(1)]],
    \\    device float* result [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]])
    \\{
    \\    result[id] = a[id] * b[id];
    \\}
    \\
    \\kernel void divide(
    \\    device const float* a [[buffer(0)]],
    \\    device const float* b [[buffer(1)]],
    \\    device float* result [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]])
    \\{
    \\    result[id] = a[id] / b[id];
    \\}
    \\
    \\kernel void scalar_multiply(
    \\    device const float* a [[buffer(0)]],
    \\    device float* result [[buffer(1)]],
    \\    constant float& scalar [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]])
    \\{
    \\    result[id] = a[id] * scalar;
    \\}
;

// Vector operations shader
pub const vector_ops_shader =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void vector_dot(
    \\    device const float* a [[buffer(0)]],
    \\    device const float* b [[buffer(1)]],
    \\    device float* result [[buffer(2)]],
    \\    device const uint& length [[buffer(3)]],
    \\    uint id [[thread_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint threadgroup_size [[threads_per_threadgroup]])
    \\{
    \\    // Each thread computes partial dot product
    \\    threadgroup float local_sum[32]; // Assuming up to 32 threads per group
    \\    
    \\    float sum = 0.0;
    \\    for (uint i = id; i < length; i += threadgroup_size) {
    \\        sum += a[i] * b[i];
    \\    }
    \\    
    \\    // Store partial sum
    \\    local_sum[tid] = sum;
    \\    
    \\    // Synchronize threadgroup
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    
    \\    // Reduce within threadgroup
    \\    for (uint s = threadgroup_size / 2; s > 0; s >>= 1) {
    \\        if (tid < s) {
    \\            local_sum[tid] += local_sum[tid + s];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    
    \\    // First thread writes the result
    \\    if (tid == 0) {
    \\        result[0] = local_sum[0];
    \\    }
    \\}
    \\
    \\kernel void vector_norm(
    \\    device const float* v [[buffer(0)]],
    \\    device float* result [[buffer(1)]],
    \\    device const uint& length [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint threadgroup_size [[threads_per_threadgroup]])
    \\{
    \\    // Each thread computes partial sum of squares
    \\    threadgroup float local_sum[32]; // Assuming up to 32 threads per group
    \\    
    \\    float sum_sq = 0.0;
    \\    for (uint i = id; i < length; i += threadgroup_size) {
    \\        sum_sq += v[i] * v[i];
    \\    }
    \\    
    \\    // Store partial sum of squares
    \\    local_sum[tid] = sum_sq;
    \\    
    \\    // Synchronize threadgroup
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    
    \\    // Reduce within threadgroup
    \\    for (uint s = threadgroup_size / 2; s > 0; s >>= 1) {
    \\        if (tid < s) {
    \\            local_sum[tid] += local_sum[tid + s];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    
    \\    // First thread writes the result
    \\    if (tid == 0) {
    \\        result[0] = sqrt(local_sum[0]);
    \\    }
    \\}
;

// Matrix-vector operations shader
pub const matrix_vector_ops_shader =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void matrix_vector_multiply(
    \\    device const float* matrix [[buffer(0)]],
    \\    device const float* vector [[buffer(1)]],
    \\    device float* result [[buffer(2)]],
    \\    device const uint& rows [[buffer(3)]],
    \\    device const uint& cols [[buffer(4)]],
    \\    uint id [[thread_position_in_grid]])
    \\{
    \\    // Each thread computes one result element (one row)
    \\    if (id >= rows) return;
    \\    
    \\    float sum = 0.0;
    \\    for (uint j = 0; j < cols; j++) {
    \\        sum += matrix[id * cols + j] * vector[j];
    \\    }
    \\    
    \\    result[id] = sum;
    \\}
;

// Matrix operations shader
pub const matrix_ops_shader =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void matrix_multiply(
    \\    device const float* a [[buffer(0)]],
    \\    device const float* b [[buffer(1)]],
    \\    device float* result [[buffer(2)]],
    \\    device const uint& M [[buffer(3)]],
    \\    device const uint& N [[buffer(4)]],
    \\    device const uint& K [[buffer(5)]],
    \\    uint2 id [[thread_position_in_grid]])
    \\{
    \\    // Each thread computes one element of the result
    \\    uint row = id.y;
    \\    uint col = id.x;
    \\    
    \\    if (row >= M || col >= N) return;
    \\    
    \\    float sum = 0.0;
    \\    for (uint k = 0; k < K; k++) {
    \\        sum += a[row * K + k] * b[k * N + col];
    \\    }
    \\    
    \\    result[row * N + col] = sum;
    \\}
    \\
    \\kernel void matrix_transpose(
    \\    device const float* matrix [[buffer(0)]],
    \\    device float* result [[buffer(1)]],
    \\    device const uint& rows [[buffer(2)]],
    \\    device const uint& cols [[buffer(3)]],
    \\    uint2 id [[thread_position_in_grid]])
    \\{
    \\    uint row = id.y;
    \\    uint col = id.x;
    \\    
    \\    if (row >= rows || col >= cols) return;
    \\    
    \\    // Transpose by swapping indices
    \\    result[col * rows + row] = matrix[row * cols + col];
    \\}
;
