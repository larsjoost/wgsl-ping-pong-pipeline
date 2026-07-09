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
    let mut pipeline = Pipeline::new().pipe(stage).build().await?;

    assert_eq!(pipeline.num_stages(), 1);
    assert_eq!(pipeline.vector_dim(), 2);
    assert_eq!(pipeline.batch_size(), 4);

    // Input: 4 vectors of 2D
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input).await?;

    // Tick once (data propagates through 1 stage)
    pipeline.tick().await?;

    // Read output
    let output = pipeline.read_output().await?;

    // Should be identical to input
    assert_eq!(output, input);
    Ok(())
}

/// Test: Two stage pipeline with identity shaders.
#[pollster::test]
async fn test_two_stage_identity() -> anyhow::Result<()> {
    let stage1 = Stage::identity("id1", 2, 4);
    let stage2 = Stage::identity("id2", 2, 4);
    let mut pipeline = Pipeline::new().pipe(stage1).pipe(stage2).build().await?;

    assert_eq!(pipeline.num_stages(), 2);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input).await?;

    // Tick twice (data needs 2 ticks for 2 stages)
    pipeline.tick().await?;
    pipeline.tick().await?;

    let output = pipeline.read_output().await?;
    assert_eq!(output, input);
    Ok(())
}

/// Test: Three stage pipeline with identity shaders.
#[pollster::test]
async fn test_three_stage_identity() -> anyhow::Result<()> {
    let stage1 = Stage::identity("id1", 2, 8);
    let stage2 = Stage::identity("id2", 2, 8);
    let stage3 = Stage::identity("id3", 2, 8);
    let mut pipeline = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .pipe(stage3)
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 3);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    pipeline.write_input(&input).await?;

    // Tick three times (data needs 3 ticks for 3 stages)
    pipeline.tick().await?;
    pipeline.tick().await?;
    pipeline.tick().await?;

    let output = pipeline.read_output().await?;
    assert_eq!(output, input);
    Ok(())
}

/// Test: Pipeline with actual compute shader (double).
#[pollster::test]
async fn test_single_stage_double() -> anyhow::Result<()> {
    let stage = Stage::new("double", DOUBLE_SHADER, 2, 4);
    let mut pipeline = Pipeline::new().pipe(stage).build().await?;

    // Input: [1,2, 3,4, 5,6, 7,8] (4 vectors of 2D)
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input).await?;

    // Tick once
    pipeline.tick().await?;

    let output = pipeline.read_output().await?;

    // Expected: [2,4, 6,8, 10,12, 14,16]
    let expected: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Two stage pipeline with identity + double.
#[pollster::test]
async fn test_two_stage_mixed() -> anyhow::Result<()> {
    let stage1 = Stage::identity("identity", 2, 4);
    let stage2 = Stage::new("double", DOUBLE_SHADER, 2, 4);
    let mut pipeline = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .build()
        .await?;

    // Input: [1,2, 3,4, 5,6, 7,8]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input).await?;

    // Tick twice (2 stages)
    pipeline.tick().await?;
    pipeline.tick().await?;

    let output = pipeline.read_output().await?;

    // Expected: doubled values [2,4, 6,8, 10,12, 14,16]
    let expected: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Validation error when vector_dim doesn't match.
#[pollster::test]
async fn test_validation_vector_dim_mismatch() -> anyhow::Result<()> {
    let stage1 = Stage::identity("s1", 2, 4);
    let stage2 = Stage::identity("s2", 3, 4); // Different vector_dim

    let result = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .build()
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("vector dimension"));
    Ok(())
}

/// Test: Validation error when batch_size doesn't match.
#[pollster::test]
async fn test_validation_batch_size_mismatch() -> anyhow::Result<()> {
    let stage1 = Stage::identity("s1", 2, 4);
    let stage2 = Stage::identity("s2", 2, 8); // Different batch_size

    let result = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .build()
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("batch size"));
    Ok(())
}

/// Test: Input size validation.
#[pollster::test]
async fn test_input_size_validation() -> anyhow::Result<()> {
    let stage = Stage::identity("id", 2, 4);
    let pipeline = Pipeline::new().pipe(stage).build().await?;

    // Try to write wrong-sized input
    let bad_input: Vec<f32> = vec![1.0, 2.0, 3.0]; // Only 3 elements, need 8
    let result = pipeline.write_input(&bad_input).await;

    assert!(result.is_err());
    Ok(())
}

/// Test: Empty pipeline error.
#[pollster::test]
async fn test_empty_pipeline() -> anyhow::Result<()> {
    let result = Pipeline::new().build().await;
    assert!(result.is_err());
    Ok(())
}

/// Test: Stage::new with custom shader.
#[pollster::test]
async fn test_stage_new() -> anyhow::Result<()> {
    let stage = Stage::new("custom", IDENTITY_2D, 2, 16);
    assert_eq!(stage.name, "custom");
    assert_eq!(stage.vector_dim, 2);
    assert_eq!(stage.batch_size, 16);
    assert_eq!(stage.element_size(), 8); // 2 * 4 bytes
    assert_eq!(stage.buffer_size(), 128); // 16 * 8 bytes
    Ok(())
}

/// Test: Stage::identity.
#[pollster::test]
async fn test_stage_identity() -> anyhow::Result<()> {
    let stage = Stage::identity("id", 3, 10);
    assert_eq!(stage.name, "id");
    assert_eq!(stage.vector_dim, 3);
    assert_eq!(stage.batch_size, 10);
    // Identity shader uses flat f32 arrays
    assert!(stage.wgsl.contains("input"));
    assert!(stage.wgsl.contains("output"));
    assert!(stage.wgsl.contains("array<f32>"));
    Ok(())
}

/// Test: Three stage identity pipeline.
#[pollster::test]
async fn test_three_stage_pipeline() -> anyhow::Result<()> {
    let stage1 = Stage::identity("stage1", 2, 16);
    let stage2 = Stage::identity("stage2", 2, 16);
    let stage3 = Stage::identity("stage3", 2, 16);
    let mut pipeline = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .pipe(stage3)
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 3);
    assert_eq!(pipeline.vector_dim(), 2);
    assert_eq!(pipeline.batch_size(), 16);
    assert_eq!(pipeline.tick_count(), 0);

    // Create input
    let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
    pipeline.write_input(&input).await?;

    // Tick 3 times
    for _ in 0..3 {
        pipeline.tick().await?;
    }

    assert_eq!(pipeline.tick_count(), 3);

    let output = pipeline.read_output().await?;
    assert_eq!(output, input);
    Ok(())
}
