//! Unit tests for the ping-pong pipeline.

use wgsl_ping_pong_pipeline::{Pipeline, Stage};

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
    let _output_tag = pipeline.tick(Some(1u64)).await?;

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
    let _output_tag1 = pipeline.tick(Some(1u64)).await?;
    let _output_tag2 = pipeline.tick(Some(1u64)).await?;

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
    let _output_tag1 = pipeline.tick(Some(1u64)).await?;
    let _output_tag2 = pipeline.tick(Some(1u64)).await?;
    let _output_tag3 = pipeline.tick(Some(1u64)).await?;

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
    let _output_tag = pipeline.tick(Some(1u64)).await?;

    let Some((_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };
    assert_eq!(output, expected);
    Ok(())
}

/// Test: Changing write data during operation.
/// In the current staggered implementation: write -> tick N times -> read -> write -> tick N times -> read
#[pollster::test]
async fn test_change_write_data_size_during_operation() -> anyhow::Result<()> {
    let stage1 = Stage::identity("id1", 2, 4);
    let stage2 = Stage::identity("id2", 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .build()
        .await?;

    // 1. Write first data (8 floats = 4 vectors of 2D)
    let input1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input1).await?;

    // 2. Tick twice for 2-stage pipeline (staggered mode)
    pipeline.tick(Some(1u64)).await?;
    pipeline.tick(Some(1u64)).await?;

    // 3. Read first data
    let Some((_tag, output1)) = pipeline.read_output().await? else {
        panic!("First output should be ready");
    };
    assert_eq!(output1, input1);

    // 4. Write second data (8 floats)
    let input2: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    pipeline.write_input(&input2).await?;

    // 5. Tick twice
    pipeline.tick(Some(2u64)).await?;
    pipeline.tick(Some(2u64)).await?;

    // 6. Read second data
    let Some((_tag, output2)) = pipeline.read_output().await? else {
        panic!("Second output should be ready");
    };
    assert_eq!(output2, input2);

    Ok(())
}

/// A test tag struct that does NOT implement Clone (to verify no cloning is needed)
#[derive(Debug, PartialEq)]
struct NonCloneTag {
    id: u64,
    data: String,
}

impl NonCloneTag {
    fn new(id: u64, data: &str) -> Self {
        Self {
            id,
            data: data.to_string(),
        }
    }
}

/// Test: Verify that tags follow data through the pipeline without requiring Clone
#[pollster::test]
async fn test_tag_follows_data_without_clone() -> anyhow::Result<()> {
    let stage = Stage::identity("identity", 2, 4);
    let mut pipeline: Pipeline<NonCloneTag> = Pipeline::new()
        .pipe(stage)
        .build()
        .await?;

    // Input data: 4 vectors of 2D (8 f32 values)
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input).await?;

    // Tick with a specific tag
    let tag1 = NonCloneTag::new(42, "first");
    pipeline.tick(Some(tag1)).await?;

    // Read output - should get the same tag back
    let Some((returned_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };

    // Verify the tag is returned correctly
    assert_eq!(returned_tag, Some(NonCloneTag::new(42, "first")));
    assert_eq!(output, input);

    // Write new data and tick with a different tag
    let input2: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    pipeline.write_input(&input2).await?;

    let tag2 = NonCloneTag::new(99, "second");
    pipeline.tick(Some(tag2)).await?;

    // Read output - should get the second tag
    let Some((returned_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after ticking");
    };

    assert_eq!(returned_tag, Some(NonCloneTag::new(99, "second")));
    assert_eq!(output, input2);

    Ok(())
}

/// Test: Verify tags follow data through multiple stages
/// In staggered mode, data takes N ticks to propagate through N stages
#[pollster::test]
async fn test_tag_follows_data_through_multiple_stages() -> anyhow::Result<()> {
    let stage1 = Stage::identity("id1", 2, 4);
    let stage2 = Stage::identity("id2", 2, 4);
    let mut pipeline: Pipeline<NonCloneTag> = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .build()
        .await?;

    // Input data
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    pipeline.write_input(&input).await?;

    // Tick with tag - in staggered mode, need N ticks for N stages
    let tag1 = NonCloneTag::new(123, "first");
    pipeline.tick(Some(tag1)).await?;
    
    // First tick: data is in stage 0, not yet at output
    // Read should return None
    let result1 = pipeline.read_output().await?;
    assert!(result1.is_none(), "Output should not be ready after first tick in 2-stage pipeline");

    // Second tick: data propagates to stage 1 and output is ready
    let tag2 = NonCloneTag::new(456, "second");
    pipeline.tick(Some(tag2)).await?;

    // Read output - should get the tag from first tick (123)
    let Some((returned_tag, output)) = pipeline.read_output().await? else {
        panic!("Output should be ready after second tick");
    };

    // Verify tag from first tick follows the data
    assert_eq!(returned_tag, Some(NonCloneTag::new(123, "first")));
    assert_eq!(output, input);

    Ok(())
}
