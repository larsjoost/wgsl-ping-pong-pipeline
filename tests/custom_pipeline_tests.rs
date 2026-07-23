//! Unit tests for custom pipeline stages.
//! 
//! These tests verify that custom stages implementing the PipelineStage trait
//! work correctly within the ping-pong pipeline.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::CommandEncoder;
use anyhow::Result;
use wgsl_ping_pong_pipeline::pipeline::{Pipeline, Stage, PipelineStage};
use wgsl_ping_pong_pipeline::wgpu_utils::ComputeContext;

/// Identity WGSL shader for custom stages
const IDENTITY_WGSL: &str = r#"
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

/// Double WGSL shader for custom stages
const DOUBLE_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input)) { return; }
    output[idx] = input[idx] * 2.0;
}
"#;

/// Triple WGSL shader for custom stages
const TRIPLE_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input)) { return; }
    output[idx] = input[idx] * 3.0;
}
"#;

/// Multiply with side input WGSL shader
/// Uses binding 0 for input, binding 1 for multiplier buffer, binding 2 for output
const MULTIPLY_WITH_SIDE_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> multiplier: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input)) { return; }
    output[idx] = input[idx] * multiplier[0];
}
"#;

/// A custom stage that uses WGSL shaders internally.
/// This demonstrates how to implement a custom stage that still uses WGSL under the hood.
#[derive(Debug)]
pub struct WgslCustomStage {
    name: String,
    wgsl: String,
    vector_dim: usize,
    batch_size: usize,
    // GPU resources created during initialization
    device: Option<Arc<wgpu::Device>>,
    compute_pipeline: Option<Arc<wgpu::ComputePipeline>>,
    bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    // For stages that use side inputs
    use_side_input: bool,
}

impl WgslCustomStage {
    /// Creates a new custom stage with the given WGSL shader.
    pub fn new(name: impl Into<String>, wgsl: impl Into<String>, vector_dim: usize, batch_size: usize) -> Self {
        Self {
            name: name.into(),
            wgsl: wgsl.into(),
            vector_dim,
            batch_size,
            device: None,
            compute_pipeline: None,
            bind_group_layout: None,
            use_side_input: false,
        }
    }

    /// Creates a new custom stage with side input support.
    pub fn with_side_input(name: impl Into<String>, wgsl: impl Into<String>, vector_dim: usize, batch_size: usize) -> Self {
        Self {
            name: name.into(),
            wgsl: wgsl.into(),
            vector_dim,
            batch_size,
            device: None,
            compute_pipeline: None,
            bind_group_layout: None,
            use_side_input: true,
        }
    }
}

impl PipelineStage for WgslCustomStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn vector_dim(&self) -> usize {
        self.vector_dim
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn side_input_names(&self) -> Vec<&str> {
        if self.use_side_input {
            vec!["multiplier"]
        } else {
            Vec::new()
        }
    }

    fn encode(
        &self,
        encoder: &mut CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        side_inputs: &HashMap<String, Arc<wgpu::Buffer>>,
    ) -> Result<()> {
        let pipeline = self.compute_pipeline.as_ref().unwrap();
        let layout = self.bind_group_layout.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();

        // Build bind group entries based on whether we use side inputs
        let entries: Vec<wgpu::BindGroupEntry> = if self.use_side_input {
            // For multiply stage: binding 0 = input, binding 1 = multiplier, binding 2 = output
            let multiplier_buffer = side_inputs.get("multiplier")
                .ok_or_else(|| anyhow::anyhow!("Side input 'multiplier' not found"))?;
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: multiplier_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ]
        } else {
            // Standard case: binding 0 = input, binding 1 = output
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ]
        };

        // Create bind group dynamically for this encode call
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", self.name)),
            layout,
            entries: &entries,
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{} Compute Pass", self.name)),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // Calculate dispatch count
        let workgroup_size = 64u32;
        let dispatch_count = (self.batch_size as u32 * self.vector_dim as u32 + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(dispatch_count, 1, 1);

        Ok(())
    }

    fn initialize(&mut self, context: &ComputeContext) -> Result<()> {
        // Store the device for later bind group creation
        self.device = Some(Arc::new(context.device.clone()));

        // Create bind group layout
        let entries: Vec<wgpu::BindGroupLayoutEntry> = if self.use_side_input {
            // For multiply stage: 3 bindings (input, multiplier, output)
            vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        } else {
            // Standard case: 2 bindings (input, output)
            vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        };

        let bgl = Arc::new(context.create_bind_group_layout(
            Some(&format!("{} Bind Group Layout", self.name)),
            &entries,
        ));

        // Create compute pipeline
        let pipeline = Arc::new(context.create_compute_pipeline(
            Some(&format!("{} Pipeline", self.name)),
            &self.wgsl,
            &[&*bgl],
        )?);

        self.bind_group_layout = Some(bgl);
        self.compute_pipeline = Some(pipeline);

        Ok(())
    }

    fn requires_initialization(&self) -> bool {
        true
    }
}

// ============================================================================
// Test Cases
// ============================================================================

/// Test: Single custom identity stage pipeline
#[pollster::test]
async fn test_single_custom_identity_stage() -> anyhow::Result<()> {
    let custom_stage = WgslCustomStage::new("custom_identity", IDENTITY_WGSL, 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(custom_stage))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 1);
    assert_eq!(pipeline.vector_dim(), 2);
    assert_eq!(pipeline.batch_size(), 4);

    // Input: 4 vectors of 2D (batch_size=4, vector_dim=2 = 8 f32 values)
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input).await?;

    // Tick once (data propagates through 1 stage)
    let _output_tag = pipeline.tick(1u64).await?;

    // Read output
    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };

    // Should be identical to input
    assert_eq!(output, input);
    Ok(())
}

/// Test: Single custom double stage pipeline
#[pollster::test]
async fn test_single_custom_double_stage() -> anyhow::Result<()> {
    let custom_stage = WgslCustomStage::new("custom_double", DOUBLE_WGSL, 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(custom_stage))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 1);
    assert_eq!(pipeline.vector_dim(), 2);
    assert_eq!(pipeline.batch_size(), 4);

    // Input: [1,2, 3,4, 5,6, 7,8] (4 vectors of 2D)
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input).await?;

    // Expected: [2,4, 6,8, 10,12, 14,16]
    let expected: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

    // Tick once to process
    let _output_tag = pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Two custom identity stages in sequence
#[pollster::test]
async fn test_two_custom_identity_stages() -> anyhow::Result<()> {
    let custom_stage1 = WgslCustomStage::new("custom_id1", IDENTITY_WGSL, 2, 4);
    let custom_stage2 = WgslCustomStage::new("custom_id2", IDENTITY_WGSL, 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(custom_stage1))
        .pipe_custom(Box::new(custom_stage2))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 2);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input).await?;

    // Tick twice (data needs 2 ticks for 2 stages)
    let _output_tag1 = pipeline.tick(1u64).await?;
    let _output_tag2 = pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, input);
    Ok(())
}

/// Test: Mixed pipeline - standard stage followed by custom stage
#[pollster::test]
async fn test_mixed_standard_then_custom() -> anyhow::Result<()> {
    // Standard identity stage + custom double stage
    let standard_stage = Stage::identity("standard_id", 2, 4);
    let custom_stage = WgslCustomStage::new("custom_double", DOUBLE_WGSL, 2, 4);
    
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe(standard_stage)
        .pipe_custom(Box::new(custom_stage))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 2);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    
    pipeline.write_input(&input).await?;

    // Tick twice (2 stages)
    let _output_tag1 = pipeline.tick(1u64).await?;
    let _output_tag2 = pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Mixed pipeline - custom stage followed by standard stage
#[pollster::test]
async fn test_mixed_custom_then_standard() -> anyhow::Result<()> {
    // Custom double stage + standard identity stage
    let custom_stage = WgslCustomStage::new("custom_double", DOUBLE_WGSL, 2, 4);
    let standard_stage = Stage::identity("standard_id", 2, 4);
    
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(custom_stage))
        .pipe(standard_stage)
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 2);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    
    pipeline.write_input(&input).await?;

    // Tick twice (2 stages)
    let _output_tag1 = pipeline.tick(1u64).await?;
    let _output_tag2 = pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Three custom stages - double, then triple, then identity
#[pollster::test]
async fn test_three_custom_stages_chain() -> anyhow::Result<()> {
    let double_stage = WgslCustomStage::new("double", DOUBLE_WGSL, 2, 4);
    let triple_stage = WgslCustomStage::new("triple", TRIPLE_WGSL, 2, 4);
    let identity_stage = WgslCustomStage::new("identity", IDENTITY_WGSL, 2, 4);
    
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(double_stage))
        .pipe_custom(Box::new(triple_stage))
        .pipe_custom(Box::new(identity_stage))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 3);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // double: [2,4,6,8,10,12,14,16]
    // triple: [6,12,18,24,30,36,42,48]
    // identity: [6,12,18,24,30,36,42,48]
    let expected: Vec<f32> = vec![6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0];
    
    pipeline.write_input(&input).await?;

    // Tick three times (3 stages)
    let _output_tag1 = pipeline.tick(1u64).await?;
    let _output_tag2 = pipeline.tick(1u64).await?;
    let _output_tag3 = pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Custom stage with side inputs (multiply with constant)
#[pollster::test]
async fn test_custom_stage_with_side_input() -> anyhow::Result<()> {
    use wgsl_ping_pong_pipeline::wgpu_utils::{stage_buffer_usages, readback_buffer_usages};
    
    // Create a shared compute context
    let context = Arc::new(ComputeContext::new_high_performance().await?);
    
    // Create multiplier buffer with value 3.0
    let multiplier_value = 3.0f32;
    let multiplier_buffer = Arc::new(context.create_buffer_with_data(
        Some("Multiplier Buffer"),
        &[multiplier_value],
        stage_buffer_usages(),
    ));
    
    // Create custom stage that uses side input
    let custom_stage = WgslCustomStage::with_side_input(
        "multiply_with_side", 
        MULTIPLY_WITH_SIDE_WGSL, 
        2, 
        4
    );
    
    // Build pipeline with custom context
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .with_context(Arc::clone(&context))
        .pipe_custom(Box::new(custom_stage))
        .build()
        .await?;
    
    // Add the side input buffer to the pipeline
    pipeline.add_side_input("multiplier", Arc::clone(&multiplier_buffer));

    assert_eq!(pipeline.num_stages(), 1);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // Each element multiplied by 3.0
    let expected: Vec<f32> = vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0];
    
    pipeline.write_input(&input).await?;

    // Tick once
    let _output_tag = pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    
    // Verify with tolerance for floating point
    for (i, (actual, expected_val)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected_val).abs() < 1e-5,
            "Mismatch at index {}: expected {}, got {}",
            i, expected_val, actual
        );
    }
    
    Ok(())
}

/// Test: Complex pipeline with 4 stages (2 standard, 2 custom)
#[pollster::test]
async fn test_complex_mixed_pipeline() -> anyhow::Result<()> {
    let standard_stage1 = Stage::identity("std_id1", 2, 4);
    let custom_stage1 = WgslCustomStage::new("custom_double", DOUBLE_WGSL, 2, 4);
    let standard_stage2 = Stage::identity("std_id2", 2, 4);
    let custom_stage2 = WgslCustomStage::new("custom_triple", TRIPLE_WGSL, 2, 4);
    
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe(standard_stage1)
        .pipe_custom(Box::new(custom_stage1))
        .pipe(standard_stage2)
        .pipe_custom(Box::new(custom_stage2))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 4);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // std_id1: [1,2,3,4,5,6,7,8]
    // custom_double: [2,4,6,8,10,12,14,16]
    // std_id2: [2,4,6,8,10,12,14,16]
    // custom_triple: [6,12,18,24,30,36,42,48]
    let expected: Vec<f32> = vec![6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0];
    
    pipeline.write_input(&input).await?;

    // Tick 4 times (4 stages)
    for _ in 0..4 {
        pipeline.tick(1u64).await?;
    }

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Custom stage with larger batch size
#[pollster::test]
async fn test_custom_stage_large_batch() -> anyhow::Result<()> {
    let batch_size = 256;
    let vector_dim = 2;
    
    let custom_stage = WgslCustomStage::new("large_double", DOUBLE_WGSL, vector_dim, batch_size);
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(custom_stage))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 1);
    assert_eq!(pipeline.batch_size(), batch_size);
    assert_eq!(pipeline.vector_dim(), vector_dim);

    // Create input with all elements set to 1.0
    let input: Vec<f32> = vec![1.0; batch_size * vector_dim];
    let expected: Vec<f32> = vec![2.0; batch_size * vector_dim];
    
    pipeline.write_input(&input).await?;
    pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Verify custom stage can be used multiple times with different configurations
#[pollster::test]
async fn test_multiple_custom_stage_instances() -> anyhow::Result<()> {
    // Create two separate instances of the same WGSL shader
    let custom_double1 = WgslCustomStage::new("double1", DOUBLE_WGSL, 2, 4);
    let custom_double2 = WgslCustomStage::new("double2", DOUBLE_WGSL, 2, 4);
    
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(custom_double1))
        .pipe_custom(Box::new(custom_double2))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 2);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // double1: [2,4,6,8,10,12,14,16]
    // double2: [4,8,12,16,20,24,28,32]
    let expected: Vec<f32> = vec![4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0];
    
    pipeline.write_input(&input).await?;

    // Tick twice
    pipeline.tick(1u64).await?;
    pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Custom stage with vector_dim = 1 (scalar values)
#[pollster::test]
async fn test_custom_stage_scalar() -> anyhow::Result<()> {
    let custom_stage = WgslCustomStage::new("scalar_double", DOUBLE_WGSL, 1, 8);
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(custom_stage))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 1);
    assert_eq!(pipeline.vector_dim(), 1);
    assert_eq!(pipeline.batch_size(), 8);

    // Input: 8 scalar values
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    
    pipeline.write_input(&input).await?;
    pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Custom stage with vector_dim = 4 (4D vectors)
#[pollster::test]
async fn test_custom_stage_4d_vectors() -> anyhow::Result<()> {
    let vector_dim = 4;
    let batch_size = 2; // 2 vectors of 4D = 8 f32 values
    
    let custom_stage = WgslCustomStage::new("4d_double", DOUBLE_WGSL, vector_dim, batch_size);
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(custom_stage))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 1);
    assert_eq!(pipeline.vector_dim(), 4);
    assert_eq!(pipeline.batch_size(), 2);

    // Input: 2 vectors of 4D = [v1x, v1y, v1z, v1w, v2x, v2y, v2z, v2w]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    
    pipeline.write_input(&input).await?;
    pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, expected);
    Ok(())
}
