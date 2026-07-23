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

    fn supports_dynamic_resizing(&self) -> bool {
        true
    }

    fn resize(&mut self, new_batch_size: usize, new_vector_dim: usize) -> Result<()> {
        self.batch_size = new_batch_size;
        self.vector_dim = new_vector_dim;
        // Recreate GPU resources with new dimensions
        // For this simple test, we just update the sizes
        // In a real implementation, we'd recreate pipelines, bind group layouts, etc.
        Ok(())
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
    use wgsl_ping_pong_pipeline::wgpu_utils::stage_buffer_usages;
    
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

/// Test: Two custom stages where submission metadata changes between ticks
/// This test demonstrates that custom stages can be part of a pipeline where
/// the submission metadata (actual_total_elements, n) changes between ticks.
/// The pipeline buffers remain the same size, but the metadata tracks the actual
/// data size in each submission.
#[pollster::test]
async fn test_two_custom_stages_with_varying_metadata() -> anyhow::Result<()> {
    let batch_size = 8;
    let vector_dim = 1;
    
    // Two custom double stages
    let stage1 = WgslCustomStage::new("stage1", DOUBLE_WGSL, vector_dim, batch_size);
    let stage2 = WgslCustomStage::new("stage2", DOUBLE_WGSL, vector_dim, batch_size);
    
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(stage1))
        .pipe_custom(Box::new(stage2))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 2);

    // Tick 1: Full buffer with metadata indicating 8 elements
    let input1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.set_input_submission_metadata(8, 8, batch_size);
    pipeline.write_input(&input1).await?;
    pipeline.tick(1u64).await?;
    pipeline.tick(1u64).await?;
    
    let Some((_tag, output1)) = pipeline.read_output().await? else {
        panic!("Output should be ready");
    };
    // [1,2,3,4,5,6,7,8] -> stage1: [2,4,6,8,10,12,14,16] -> stage2: [4,8,12,16,20,24,28,32]
    assert_eq!(output1, vec![4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]);

    // Tick 2: Different data, metadata still 8 elements
    let input2: Vec<f32> = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    pipeline.set_input_submission_metadata(8, 8, batch_size);
    pipeline.write_input(&input2).await?;
    pipeline.tick(2u64).await?;
    pipeline.tick(2u64).await?;
    
    let Some((_tag, output2)) = pipeline.read_output().await? else {
        panic!("Output should be ready");
    };
    // All 10s -> stage1: all 20s -> stage2: all 40s
    assert_eq!(output2, vec![40.0; 8]);

    // Tick 3: Metadata indicates only 4 actual elements (but buffer still has 8)
    // The pipeline still processes all 8 buffer elements, but the metadata tracks that
    // only the first 4 are "actual" data
    let input3: Vec<f32> = vec![100.0, 200.0, 300.0, 400.0, 0.0, 0.0, 0.0, 0.0];
    pipeline.set_input_submission_metadata(4, 4, batch_size);
    pipeline.write_input(&input3).await?;
    pipeline.tick(3u64).await?;
    pipeline.tick(3u64).await?;
    
    let Some((_tag, output3)) = pipeline.read_output().await? else {
        panic!("Output should be ready");
    };
    // [100,200,300,400,0,0,0,0] -> stage1: [200,400,600,800,0,0,0,0] -> stage2: [400,800,1200,1600,0,0,0,0]
    assert_eq!(output3[0..4], vec![400.0, 800.0, 1200.0, 1600.0]);
    assert_eq!(output3[4..8], vec![0.0; 4]);
    
    Ok(())
}

/// Test: Two custom stages with dynamic buffer growth
/// This test demonstrates that the pipeline can dynamically resize its buffers
/// when the data size increases beyond the initial buffer capacity.
#[pollster::test]
async fn test_two_custom_stages_dynamic_buffer_growth() -> anyhow::Result<()> {
    let initial_batch_size = 4;
    let vector_dim = 1;
    
    // Create two custom double stages
    let stage1 = WgslCustomStage::new("stage1", DOUBLE_WGSL, vector_dim, initial_batch_size);
    let stage2 = WgslCustomStage::new("stage2", DOUBLE_WGSL, vector_dim, initial_batch_size);
    
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(stage1))
        .pipe_custom(Box::new(stage2))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 2);
    assert_eq!(pipeline.batch_size(), initial_batch_size);

    // First: Process with initial size (4 elements)
    let input1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    pipeline.set_input_submission_metadata(4, 4, initial_batch_size);
    pipeline.write_input(&input1).await?;
    pipeline.tick(1u64).await?;
    pipeline.tick(1u64).await?;
    
    let Some((_tag, output1)) = pipeline.read_output().await? else {
        panic!("Output should be ready");
    };
    // [1,2,3,4] -> stage1: [2,4,6,8] -> stage2: [4,8,12,16]
    assert_eq!(output1, vec![4.0, 8.0, 12.0, 16.0]);

    // Now resize the pipeline to handle larger data (8 elements)
    let new_batch_size = 8;
    pipeline.resize(new_batch_size).await?;
    assert_eq!(pipeline.batch_size(), new_batch_size);
    
    // Second: Process with new larger size (8 elements)
    let input2: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    pipeline.set_input_submission_metadata(8, 8, new_batch_size);
    pipeline.write_input(&input2).await?;
    pipeline.tick(2u64).await?;
    pipeline.tick(2u64).await?;
    
    let Some((_tag, output2)) = pipeline.read_output().await? else {
        panic!("Output should be ready after resize");
    };
    // [10,20,30,40,50,60,70,80] -> stage1: [20,40,60,80,100,120,140,160] -> stage2: [40,80,120,160,200,240,280,320]
    assert_eq!(output2, vec![40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0, 320.0]);

    // Resize again to handle even larger data (16 elements)
    let larger_batch_size = 16;
    pipeline.resize(larger_batch_size).await?;
    assert_eq!(pipeline.batch_size(), larger_batch_size);
    
    // Third: Process with even larger size (16 elements)
    let input3: Vec<f32> = vec![1.0; 16];
    pipeline.set_input_submission_metadata(16, 16, larger_batch_size);
    pipeline.write_input(&input3).await?;
    pipeline.tick(3u64).await?;
    pipeline.tick(3u64).await?;
    
    let Some((_tag, output3)) = pipeline.read_output().await? else {
        panic!("Output should be ready after second resize");
    };
    // All 1s -> stage1: all 2s -> stage2: all 4s
    assert_eq!(output3, vec![4.0; 16]);
    
    Ok(())
}

/// Test: Two custom stages with per-stage buffer resizing
/// This demonstrates that the VariableSizePipeline can resize individual stage buffers.
#[pollster::test]
async fn test_two_custom_stages_selective_resize() -> anyhow::Result<()> {
    use wgsl_ping_pong_pipeline::pipeline::variable_size::{VariableSizePipeline, VariableSizePipelineBuilder};
    
    let vector_dim = 1;
    
    // Create two custom double stages with initial size 4
    let stage1 = WgslCustomStage::new("stage1", DOUBLE_WGSL, vector_dim, 4);
    let stage2 = WgslCustomStage::new("stage2", DOUBLE_WGSL, vector_dim, 4);
    
    let mut pipeline: VariableSizePipeline<u64> = VariableSizePipelineBuilder::new()
        .pipe_custom_with_size(Box::new(stage1), 4)
        .pipe_custom_with_size(Box::new(stage2), 4)
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 2);

    // Process with initial size (4 elements)
    let input1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    pipeline.write_input(&input1).await?;
    pipeline.tick(1u64).await?;
    pipeline.tick(1u64).await?;
    
    let Some((_tag, output1)) = pipeline.read_output().await? else {
        panic!("Output should be ready");
    };
    assert_eq!(output1, vec![4.0, 8.0, 12.0, 16.0]);

    // Use resize_stage_both to resize both buffers in a pair at once
    // This is simpler and ensures both buffers match
    pipeline.resize_stage_both(2, 8).await?;  // Stage 2 = stage 1's output buffers
    pipeline.resize_stage_both(1, 8).await?;  // Stage 1 = stage 0's output buffers  
    pipeline.resize_stage_both(0, 8).await?;  // Stage 0 = input buffers
    
    // Verify all buffers are now size 8
    let (input_a, input_b) = pipeline.get_stage_buffer_sizes(0);
    let (stage1_a, stage1_b) = pipeline.get_stage_buffer_sizes(1);
    let (stage2_a, stage2_b) = pipeline.get_stage_buffer_sizes(2);
    assert_eq!(input_a, 8); assert_eq!(input_b, 8);
    assert_eq!(stage1_a, 8); assert_eq!(stage1_b, 8);
    assert_eq!(stage2_a, 8); assert_eq!(stage2_b, 8);
    
    // Now process with full 8 elements
    let input2: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input2).await?;
    pipeline.tick(2u64).await?;
    pipeline.tick(2u64).await?;
    
    let Some((_tag, output2)) = pipeline.read_output().await? else {
        panic!("Output should be ready");
    };
    
    // All 8 elements processed correctly
    assert_eq!(output2, vec![4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]);
    
    Ok(())
}

/// Test: Verify that resize_stage() only resizes ONE buffer in the ping-pong pair
/// This test demonstrates proper ping-pong behavior where only the non-writing buffer is resized.
#[pollster::test]
async fn test_selective_resize_one_buffer_only() -> anyhow::Result<()> {
    use wgsl_ping_pong_pipeline::pipeline::variable_size::{VariableSizePipeline, VariableSizePipelineBuilder};
    
    let vector_dim = 1;
    
    // Create two custom double stages with initial size 4
    let stage1 = WgslCustomStage::new("stage1", DOUBLE_WGSL, vector_dim, 4);
    let stage2 = WgslCustomStage::new("stage2", DOUBLE_WGSL, vector_dim, 4);
    
    let mut pipeline: VariableSizePipeline<u64> = VariableSizePipelineBuilder::new()
        .pipe_custom_with_size(Box::new(stage1), 4)
        .pipe_custom_with_size(Box::new(stage2), 4)
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 2);

    // Verify initial buffer sizes - all should be 4
    let (input_a, input_b) = pipeline.get_stage_buffer_sizes(0);
    let (stage0_out_a, stage0_out_b) = pipeline.get_stage_buffer_sizes(1);
    let (stage1_out_a, stage1_out_b) = pipeline.get_stage_buffer_sizes(2);
    assert_eq!(input_a, 4); assert_eq!(input_b, 4);
    assert_eq!(stage0_out_a, 4); assert_eq!(stage0_out_b, 4);
    assert_eq!(stage1_out_a, 4); assert_eq!(stage1_out_b, 4);

    // Do one tick to put pipeline in a known state
    // After one tick, current_input_write_index will be 1 (flipped from 0)
    // and current_output_indices will be [1, 1] (flipped from [0, 0])
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    pipeline.write_input(&input).await?;
    pipeline.tick(1u64).await?;

    // Now resize stage 1's output buffers (stage_idx=2) to size 8
    // This should only resize ONE of the two buffers (the one NOT being written to)
    pipeline.resize_stage(2, 8).await?;

    // Verify that only ONE buffer was resized
    let (stage1_out_a, stage1_out_b) = pipeline.get_stage_buffer_sizes(2);
    // One should be 8 (the resized one), the other should still be 4 (the old one)
    let sizes_match = (stage1_out_a == 8 && stage1_out_b == 4) || (stage1_out_a == 4 && stage1_out_b == 8);
    assert!(sizes_match, "Expected one buffer to be size 8 and the other size 4, got ({}, {})", stage1_out_a, stage1_out_b);

    // Verify other buffers were NOT affected
    let (input_a, input_b) = pipeline.get_stage_buffer_sizes(0);
    let (stage0_out_a, stage0_out_b) = pipeline.get_stage_buffer_sizes(1);
    assert_eq!(input_a, 4); assert_eq!(input_b, 4);
    assert_eq!(stage0_out_a, 4); assert_eq!(stage0_out_b, 4);

    // Do another tick - the pipeline should still work
    // The metadata for stage 1's buffers will now have the resized buffer available
    pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready");
    };
    // Output should still be correct: [1,2,3,4] -> stage1: [2,4,6,8] -> stage2: [4,8,12,16]
    assert_eq!(output, vec![4.0, 8.0, 12.0, 16.0]);

    Ok(())
}

/// Test: Changing write data size during operation with Custom stages.
/// In immediate mode: write data -> tick -> read output -> write new data -> tick -> read new output
#[pollster::test]
async fn test_custom_stage_change_write_data_size_during_operation() -> anyhow::Result<()> {
    let stage1 = WgslCustomStage::new("custom_stage1", DOUBLE_WGSL, 2, 4);
    let stage2 = WgslCustomStage::new("custom_stage2", DOUBLE_WGSL, 2, 4);
    
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe_custom(Box::new(stage1))
        .pipe_custom(Box::new(stage2))
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 2);

    // 1. Write data with first size (4 floats = 2 vectors of 2D)
    let input1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    pipeline.write_input(&input1).await?;

    // 2. Tick
    pipeline.tick(1u64).await?;

    // 3. Read first data immediately: [1,2,3,4] -> stage1: [2,4,6,8] -> stage2: [4,8,12,16]
    let Some((_tag, output1)) = pipeline.read_output().await? else {
        panic!("First output should be ready");
    };
    assert_eq!(output1, vec![4.0, 8.0, 12.0, 16.0]);

    // 4. Write data with another size (12 floats = 6 vectors of 2D)
    let input2: Vec<f32> = vec![
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
    ];
    pipeline.write_input(&input2).await?;

    // 5. Tick
    pipeline.tick(2u64).await?;

    // 6. Read second data immediately: [10,20,...,120] -> double twice -> [40,80,...,480]
    let Some((_tag, output2)) = pipeline.read_output().await? else {
        panic!("Second output should be ready");
    };
    let expected2: Vec<f32> = input2.iter().map(|x| x * 4.0).collect();
    assert_eq!(output2, expected2);

    Ok(())
}


