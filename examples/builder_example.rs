//! Example demonstrating the pipeline builder pattern.

use wgsl_ping_pong_pipeline::{Pipeline, Stage};

/// Identity shader (flat array).
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

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("Building pipeline with builder pattern...");

    // Create stages with the same vector_dim and batch_size
    let stage1 = Stage::new("stage1", IDENTITY_2D, 2, 1024);
    let stage2 = Stage::new("stage2", IDENTITY_2D, 2, 1024);
    let stage3 = Stage::new("stage3", IDENTITY_2D, 2, 1024);

    // Use the builder pattern: pipe().pipe().build()
    let mut pipeline: Pipeline<u64> = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .pipe(stage3)
        .build()
        .await?;

    println!("Pipeline built successfully!");
    println!("  Stages: {}", pipeline.num_stages());
    println!("  Vector dimension: {}", pipeline.vector_dim());
    println!("  Batch size: {}", pipeline.batch_size());

    // Create input data
    let input: Vec<f32> = (0..1024 * 2).map(|i| i as f32).collect();

    // Process the pipeline - only the last call returns Some with output
    let mut result = None;
    for i in 0..pipeline.num_stages() {
        if i == 0 {
            // First call: write input and advance pipeline
            result = pipeline.process(Some(&input), 1u64).await?;
        } else {
            // Subsequent calls: just advance pipeline
            result = pipeline.process(None, 1u64).await?;
        }
        println!("Process {}", i + 1);
    }

    // The last call should return Some with output
    let Some((output_tag, output)) = result else {
        println!("Output not ready yet");
        return Ok(());
    };
    println!(
        "Output read: {} elements, tag: {:?}",
        output.len(),
        output_tag
    );
    println!(
        "Output matches input (first {} elements): {}",
        input.len(),
        output[..input.len()] == input
    );

    Ok(())
}
