//! Tests for the new process() function that merges tick() and read_output().

use wgsl_ping_pong_pipeline::{Pipeline, Stage};

/// Test: Single stage pipeline with process() function.
#[pollster::test]
async fn test_single_stage_process() -> anyhow::Result<()> {
    let stage = Stage::identity("identity", 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new().pipe(stage).build().await?;

    // Input: 4 vectors of 2D (batch_size=4, vector_dim=2 = 8 f32 values)
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Process once (data propagates through 1 stage and output is available)
    let Some((_tag, output)) = pipeline.process(Some(&input), 1u64).await? else {
        panic!("Output should be ready after first process in 1-stage pipeline");
    };

    // Should be identical to input
    assert_eq!(output, input);
    Ok(())
}

/// Test: Two stage pipeline with process() function.
#[pollster::test]
async fn test_two_stage_process() -> anyhow::Result<()> {
    let stage1 = Stage::identity("id1", 2, 4);
    let stage2 = Stage::identity("id2", 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new().pipe(stage1).pipe(stage2).build().await?;

    assert_eq!(pipeline.num_stages(), 2);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // First process: returns None (data hasn't propagated through all stages yet)
    let result1 = pipeline.process(Some(&input), 1u64).await?;
    assert!(
        result1.is_none(),
        "First process should return None for 2-stage pipeline"
    );

    // Second process: returns Some with output
    let Some((_tag, output)) = pipeline.process(None, 1u64).await? else {
        panic!("Second process should return Some for 2-stage pipeline");
    };
    assert_eq!(output, input);
    Ok(())
}

/// Test: Three stage pipeline with process() function.
#[pollster::test]
async fn test_three_stage_process() -> anyhow::Result<()> {
    let stage1 = Stage::identity("id1", 2, 4);
    let stage2 = Stage::identity("id2", 2, 4);
    let stage3 = Stage::identity("id3", 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .pipe(stage3)
        .build()
        .await?;

    assert_eq!(pipeline.num_stages(), 3);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // First process: returns None
    let result1 = pipeline.process(Some(&input), 1u64).await?;
    assert!(
        result1.is_none(),
        "First process should return None for 3-stage pipeline"
    );

    // Second process: returns None
    let result2 = pipeline.process(None, 1u64).await?;
    assert!(
        result2.is_none(),
        "Second process should return None for 3-stage pipeline"
    );

    // Third process: returns Some with output
    let Some((_tag, output)) = pipeline.process(None, 1u64).await? else {
        panic!("Third process should return Some for 3-stage pipeline");
    };
    assert_eq!(output, input);
    Ok(())
}

/// Test: Process with different tags.
#[pollster::test]
async fn test_process_tag_propagation() -> anyhow::Result<()> {
    let stage1 = Stage::identity("id1", 2, 4);
    let stage2 = Stage::identity("id2", 2, 4);
    let mut pipeline: Pipeline<u64> = Pipeline::new().pipe(stage1).pipe(stage2).build().await?;

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let tag1 = 123u64;
    let tag2 = 456u64;

    // First process with tag1: returns None
    pipeline.process(Some(&input), tag1).await?;

    // Second process with tag2: returns Some with tag1 (from first process)
    let Some((returned_tag, output)) = pipeline.process(None, tag2).await? else {
        panic!("Should have output after second process");
    };

    // The returned tag should be from the first process call (delay line behavior)
    assert_eq!(returned_tag, tag1);
    assert_eq!(output, input);

    Ok(())
}
