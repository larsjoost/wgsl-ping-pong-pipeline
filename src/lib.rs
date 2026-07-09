//! WGSL Ping-Pong Pipeline
//!
//! A high-throughput pipelined compute cascade using wgpu with isolated per-stage
//! ping-pong buffers for maximum parallel throughput.
//!
//! ## Usage
//!
//! ```no_run
//! use wgsl_ping_pong_pipeline::{Pipeline, Stage};
//!
//! // Simple identity shader
//! const IDENTITY_SHADER: &str = r#"
//! @group(0) @binding(0)
//! var<storage, read> input: array<f32>;
//! @group(0) @binding(1)
//! var<storage, read_write> output: array<f32>;
//! @compute @workgroup_size(64)
//! fn main(@builtin(global_invocation_id) id: vec3<u32>) {
//!     let idx = id.x;
//!     if (idx >= arrayLength(&input)) { return; }
//!     output[idx] = input[idx];
//! }
//! "#;
//!
//! #[pollster::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create stages with shaders
//!     let stage1 = Stage::new("stage1", IDENTITY_SHADER, 4, 1024);
//!     let stage2 = Stage::new("stage2", IDENTITY_SHADER, 4, 1024);
//!     let stage3 = Stage::new("stage3", IDENTITY_SHADER, 4, 1024);
//!
//!     // Build the pipeline using the builder pattern
//!     let mut pipeline = Pipeline::new()
//!         .pipe(stage1)
//!         .pipe(stage2)
//!         .pipe(stage3)
//!         .build()
//!         .await?;
//!
//!     // Write input data
//!     let input: Vec<f32> = vec![0.0; 1024 * 4];
//!     pipeline.write_input(&input).await?;
//!
//!     // Advance pipeline by 3 ticks (data propagates through 3 stages)
//!     pipeline.tick().await?;
//!     pipeline.tick().await?;
//!     pipeline.tick().await?;
//!
//!     Ok(())
//! }
//! ```

//#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod pipeline;
pub mod wgpu_utils;

pub use pipeline::{Pipeline, PipelineBuilder, Stage};
pub use pipeline::pipeline_stage::{PipelineStage, StageConfig};
pub use wgpu_utils::ComputeContext;
