# wgsl-ping-pong-pipeline

A high-throughput pipelined compute cascade using WGSL and wgpu with isolated per-stage ping-pong buffers for maximum parallel throughput.

## Usage in gpu_pipeline

This pipeline is used as the default implementation in the `gpu_pipeline` crate for FFT-based convolution operations. See the `gpu_pipeline` crate for a high-level API that uses this pipeline internally.

### Integration Example

To use this pipeline directly for FFT-based convolution (FFT -> Multiply -> IFFT):

```rust
use wgsl_ping_pong_pipeline::pipeline::{Pipeline, StageConfig};
use wgsl_fft::ping_pong_integration::{FftPipelineStage, MultiplyPipelineStage};

let n = 1024;
let batch_size = 1;

let pipeline = Pipeline::new()
    .pipe_config(StageConfig::Custom(Box::new(FftPipelineStage::forward(n, batch_size))))
    .pipe_config(StageConfig::Custom(Box::new(MultiplyPipelineStage::new(n, batch_size))))
    .pipe_config(StageConfig::Custom(Box::new(FftPipelineStage::inverse(n, batch_size))))
    .build()
    .await?;

// Add B input as side input for multiplication
let b_buffer: Arc<wgpu::Buffer> = ...;
pipeline.add_side_input("input_b", Arc::clone(&b_buffer));

// Process through all stages
// For a 3-stage pipeline with data, the first 2 calls return None, the 3rd returns Some(output)
let a_data: Vec<f32> = ...;
pipeline.process(Some(&a_data), 1u64).await?; // Returns None (data written, not ready yet)
pipeline.process(None, 2u64).await?; // Returns None (advancing)
let Some((_tag, output)) = pipeline.process(None, 3u64).await? else {
    panic!("Output should be ready after 3 calls for 3-stage pipeline");
};
```

**Important**: For convolution operations, ensure that input B is in the frequency domain before being passed to the multiply stage. You may need to preprocess B with FFT depending on your use case.

## Overview

This crate provides a generic, extensible pipeline framework for GPU compute workloads. It implements the **ping-pong buffering pattern** to enable overlapping computation across multiple stages, maximizing GPU utilization.

### Key Features

- **Ping-Pong Buffering**: Each stage has isolated input/output buffers that alternate (ping-pong), allowing data to flow continuously through the pipeline
- **Generic Design**: Works with any WGSL compute shader - no domain-specific knowledge required
- **Custom Stages**: Support for custom `PipelineStage` implementations via the trait system
- **Side Inputs**: Register additional buffers as side inputs for stages that need external data (e.g., pre-computed FFT kernels)
- **Shared Context**: Optionally share a `ComputeContext` across pipeline and external GPU resources

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
wgsl-ping-pong-pipeline = { git = "https://github.com/larsjoost/wgsl-ping-pong-pipeline" }
```

## Quick Start

### Basic Usage with Standard Stages

```rust
use wgsl_ping_pong_pipeline::{Pipeline, Stage};

// Define a simple identity shader
const IDENTITY_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input)) { return; }
    output[idx] = input[idx];
}
"#;

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    // Create stages with the same vector_dim and batch_size
    let stage1 = Stage::new("stage1", IDENTITY_SHADER, 2, 1024);
    let stage2 = Stage::new("stage2", IDENTITY_SHADER, 2, 1024);
    
    // Build the pipeline using the builder pattern
    let mut pipeline = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .build()
        .await?;

    // Process input data through pipeline
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    pipeline.process(Some(&input), 1u64).await?; // Returns None for 2-stage (data written)
    let Some((_tag, output)) = pipeline.process(None, 2u64).await? else {
        panic!("Output should be ready after 2nd call for 2-stage pipeline");
    };
    
    Ok(())
}
```

### Using Custom Stages

The pipeline supports custom stages via the `PipelineStage` trait. This allows integration with domain-specific libraries like [wgsl-fft](https://github.com/larsjoost/wgsl-fft):

```rust
use std::sync::Arc;
use wgsl_ping_pong_pipeline::{Pipeline, StageConfig, PipelineStage};
use wgsl_fft::ping_pong_integration::{FftPipelineStage, MultiplyPipelineStage};

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    // Build an FFT-based convolution pipeline
    let mut pipeline = Pipeline::new()
        .pipe_config(StageConfig::Custom(
            Box::new(FftPipelineStage::forward(1024, 1))
        ))
        .pipe_config(StageConfig::Custom(
            Box::new(MultiplyPipelineStage::new(1024, 1))
        ))
        .pipe_config(StageConfig::Custom(
            Box::new(FftPipelineStage::inverse(1024, 1))
        ))
        .build()
        .await?;

    // Register a side input for the multiply stage
    // let fft_b_buffer: Arc<wgpu::Buffer> = ...;
    // pipeline.add_side_input("input_b", Arc::clone(&fft_b_buffer));

    Ok(())
}
```

### Sharing GPU Context

To share a `ComputeContext` between the pipeline and external GPU resources (e.g., pre-allocated buffers):

```rust
use std::sync::Arc;
use wgsl_ping_pong_pipeline::{Pipeline, ComputeContext, Stage};

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    // Create a shared context
    let context = Arc::new(ComputeContext::new_high_performance().await?);
    
    // Use the shared context for the pipeline
    let mut pipeline = Pipeline::new()
        .with_context(Arc::clone(&context))
        .pipe(Stage::new("stage1", SHADER, 2, 1024))
        .build()
        .await?;

    // Create buffers using the same context's device
    let buffer = Arc::new(context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("My Buffer"),
        size: 1024 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    Ok(())
}
```

## Architecture

### Pipeline Structure

The pipeline consists of multiple **stages**, each with:
- An input buffer (read from previous stage)
- An output buffer pair (ping-pong buffers)
- A compute pipeline for GPU execution

```
Input -> [Stage 0] -> [Stage 1] -> [Stage 2] -> Output
           |            |            |
          out_a       out_a       out_a
          out_b       out_b       out_b
```

### Ping-Pong Pattern

Each stage has two output buffers that alternate:
- **Tick 0**: Write to buffer A
- **Tick 1**: Write to buffer B (read from previous stage's buffer A)
- **Tick 2**: Write to buffer A (read from previous stage's buffer B)
- And so on...

This allows the GPU to process multiple frames simultaneously, maximizing throughput.

## API Reference

### Core Types

- `Pipeline` - The main pipeline structure
- `PipelineBuilder` - Builder for creating pipelines
- `Stage` - A standard compute stage with WGSL shader
- `StageConfig` - Configuration for a stage (Standard or Custom)
- `PipelineStage` - Trait for custom stage implementations
- `ComputeContext` - Wrapper for wgpu Device and Queue

### Key Methods

#### Pipeline
- `Pipeline::new()` - Create a new pipeline builder
- `pipeline.process(data, tag)` - Write input data and advance the pipeline by one step, reading output when available. Returns `Option<(T, Vec<f32>)>`. For N-stage pipelines, the first N-1 calls after writing data return `None`, the Nth call returns `Some((tag, output))` with the delayed tag. Call with `None` for data to advance without writing new input.
- `pipeline.add_side_input(name, buffer)` - Add a side input buffer
- `pipeline.resize(new_batch_size)` - Resize all buffers

#### PipelineBuilder
- `.pipe(stage)` - Add a standard stage
- `.pipe_config(config)` - Add a stage configuration
- `.pipe_custom(stage)` - Add a custom stage
- `.with_context(context)` - Set a shared ComputeContext
- `.build()` - Build the pipeline

#### Custom Stages
Implement the `PipelineStage` trait:

```rust
use std::collections::HashMap;
use std::sync::Arc;
use wgsl_ping_pong_pipeline::{PipelineStage, ComputeContext};
use anyhow::Result;

struct MyCustomStage {
    // Your fields here
}

impl PipelineStage for MyCustomStage {
    fn name(&self) -> &str {
        "my_custom_stage"
    }
    
    fn vector_dim(&self) -> usize {
        2  // e.g., for vec2<f32> (complex numbers)
    }
    
    fn batch_size(&self) -> usize {
        1024  // Number of elements per batch
    }
    
    fn side_input_names(&self) -> Vec<&str> {
        vec!["input_b"]  // Names of required side inputs
    }
    
    fn requires_initialization(&self) -> bool {
        true  // Set to true if you need to initialize GPU resources
    }
    
    fn initialize(&mut self, context: &ComputeContext) -> Result<()> {
        // Initialize GPU resources using context.device and context.queue
        Ok(())
    }
    
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        side_inputs: &HashMap<String, Arc<wgpu::Buffer>>,
    ) -> Result<()> {
        // Encode your compute pass here
        // Use encoder.begin_compute_pass(), set_pipeline(), dispatch_workgroups(), etc.
        Ok(())
    }
}
```

## Examples

- `examples/builder_example.rs` - Basic pipeline with identity shaders

## Integration with wgsl-fft

For FFT-based signal processing, use the `wgsl-fft` crate with the `ping_pong` feature:

```toml
[dependencies]
wgsl-fft = { version = "0.4", features = ["ping_pong"] }
```

This provides ready-to-use `FftPipelineStage` and `MultiplyPipelineStage` implementations.

## Requirements

- Rust 2024 edition
- wgpu 29.0+
- A compatible GPU (Vulkan, Metal, DirectX, or browser WebGPU)

## License

MIT License
