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
    let mut pipeline = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .pipe(stage3)
        .build()
        .await?;

    println!("✓ Pipeline built successfully!");
    println!("  Stages: {}", pipeline.num_stages());
    println!("  Vector dimension: {}", pipeline.vector_dim());
    println!("  Batch size: {}", pipeline.batch_size());

    // Create input data
    let input: Vec<f32> = (0..1024 * 2).map(|i| i as f32).collect();

    // Write input
    pipeline.write_input(&input).await?;
    println!("✓ Input written");

    // Tick the pipeline
    for i in 0..pipeline.num_stages() {
        pipeline.tick().await?;
        println!("✓ Tick {}", i + 1);
    }

    // Read output
    let output = pipeline.read_output().await?;
    println!("✓ Output read ({} elements)", output.len());

    // Since we used identity shaders, output should equal input
    if output == input {
        println!("✓ Pipeline works correctly (identity shaders)");
    } else {
        println!("✗ Output differs from input");
    }

    Ok(())
}
