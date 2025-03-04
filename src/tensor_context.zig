pub const TensorContext = struct {
    backend: Backend = .cpu,
};

pub const Backend = enum {
    cpu,
    gpu,
};
