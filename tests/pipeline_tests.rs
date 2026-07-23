//! Unit tests for the ping-pong pipeline.

use wgsl_ping_pong_pipeline::{Pipeline, Stage};

/// Identity shader for 2D vectors (flat array).
const IDENTITY_2D: &str = r#"
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

/// Shader that doubles each element (flat array).
const DOUBLE_SHADER: &str = r#"
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

/// Test: Single stage pipeline with identity shader.
#[pollster::test]
async fn test_single_stage_identity() -> anyhow::Result<()> {
    let stage = Stage::identity("identity", 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new().pipe(stage).build().await?;

    assert_eq!(pipeline.num_stages(), 1);
    assert_eq!(pipeline.vector_dim(), 2);
    assert_eq!(pipeline.batch_size(), 4);

    // Input: 4 vectors of 2D (batch_size=4, vector_dim=2 = 8 f32 values)
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input).await?;

    // Tick once (data propagates through 1 stage)
    let _output_tag = pipeline.tick(1u64).await?;

    // Read output (returns Option<(tag, data)>)
    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };

    // Should be identical to input
    assert_eq!(output, input);
    Ok(())
}

/// Test: Two stage pipeline with identity shaders.
#[pollster::test]
async fn test_two_stage_identity() -> anyhow::Result<()> {
    let stage1 = Stage::identity("id1", 2, 4);
    let stage2 = Stage::identity("id2", 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new().pipe(stage1).pipe(stage2).build().await?;

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

/// Test: Three stage pipeline with identity shaders.
#[pollster::test]
async fn test_three_stage_identity() -> anyhow::Result<()> {
    let stage1 = Stage::identity("id1", 2, 8);
    let stage2 = Stage::identity("id2", 2, 8);
    let stage3 = Stage::identity("id3", 2, 8);
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .pipe(stage3)
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 3);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    pipeline.write_input(&input).await?;

    // Tick three times (data needs 3 ticks for 3 stages)
    let _output_tag1 = pipeline.tick(1u64).await?;
    let _output_tag2 = pipeline.tick(1u64).await?;
    let _output_tag3 = pipeline.tick(1u64).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, input);
    Ok(())
}

/// Test: Pipeline with actual compute shader (double).
#[pollster::test]
async fn test_single_stage_double() -> anyhow::Result<()> {
    let stage = Stage::new("double", DOUBLE_SHADER, 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new().pipe(stage).build().await?;

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

/// Test: Changing write data size during operation.
/// Sequence: write data with one size -> tick -> write data with another size -> tick -> read first data -> tick -> read second data
#[pollster::test]
async fn test_change_write_data_size_during_operation() -> anyhow::Result<()> {
    let stage1 = Stage::identity("id1", 2, 4);
    let stage2 = Stage::identity("id2", 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .build()
        .await?;

    // 1. Write data with first size (4 floats = 2 vectors of 2D)
    let input1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    pipeline.write_input(&input1).await?;

    // 2. Tick
    pipeline.tick(1u64).await?;

    // 3. Write data with another size (12 floats = 6 vectors of 2D)
    let input2: Vec<f32> = vec![
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
    ];
    pipeline.write_input(&input2).await?;

    // 4. Tick
    pipeline.tick(2u64).await?;

    // 5. Read first data
    let Some((_tag, output1)) = pipeline.read_output().await? else {
        panic!("First output should be ready");
    };
    assert_eq!(output1, input1);

    // 6. Tick
    pipeline.tick(3u64).await?;

    // 7. Read second data
    let Some((_tag, output2)) = pipeline.read_output().await? else {
        panic!("Second output should be ready");
    };
    assert_eq!(output2, input2);

    Ok(())
}
