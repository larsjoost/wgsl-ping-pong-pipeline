//! Simple example demonstrating the ping-pong pipeline with identity stages.

use wgsl_ping_pong_pipeline::{Pipeline, Stage};

/// A simple compute shader that doubles each element (flat array).
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

/// A simple compute shader that adds 1.0 to each element (flat array).
const ADD_ONE_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input)) { return; }
    output[idx] = input[idx] + 1.0;
}
"#;

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Create 3 stages: identity -> double -> add_one
    // All must have the same vector_dim (2) and batch_size (8)
    let stage1 = Stage::identity("identity", 2, 8);
    let stage2 = Stage::new("double", DOUBLE_SHADER, 2, 8);
    let stage3 = Stage::new("add_one", ADD_ONE_SHADER, 2, 8);

    // Build the pipeline using the builder pattern
    let mut pipeline = Pipeline::new()
        .pipe(stage1)
        .pipe(stage2)
        .pipe(stage3)
        .build()
        .await?;

    println!("Pipeline created with {} stages", pipeline.num_stages());
    println!("Vector dimension: {}", pipeline.vector_dim());
    println!("Batch size: {}", pipeline.batch_size());

    // Create input data: 8 vectors of 2D (x, y)
    // Input: [1, 2, 3, 4, 5, 6, 7, 8] for x, [10, 20, 30, 40, 50, 60, 70, 80] for y
    let input: Vec<f32> = vec![
        1.0, 10.0, // Vector 0: x=1, y=10
        2.0, 20.0, // Vector 1: x=2, y=20
        3.0, 30.0, // Vector 2: x=3, y=30
        4.0, 40.0, // Vector 3: x=4, y=40
        5.0, 50.0, // Vector 4: x=5, y=50
        6.0, 60.0, // Vector 5: x=6, y=60
        7.0, 70.0, // Vector 6: x=7, y=70
        8.0, 80.0, // Vector 7: x=8, y=80
    ];

    // Process data through the pipeline (first 2 calls return None, 3rd returns output)
    // We use a simple tag (u64) to track the data flow
    let tag = 1u64;
    pipeline.process(Some(&input), tag).await?; // Call 1: returns None (data in stage 1)
    println!("Process 1: Data in stage 1");

    pipeline.process(None, 2u64).await?; // Call 2: returns None (data in stage 2)
    println!("Process 2: Data in stage 2");

    // Call 3: returns Some with output (data has propagated through all 3 stages)
    let Some((output_tag, output)) = pipeline.process(None, 3u64).await? else {
        println!("Output not ready yet");
        return Ok(());
    };
    println!(
        "Process 3: Data in stage 3 (output ready), output tag: {:?}",
        output_tag
    );
    println!("Output: {:?}", output);

    // Expected output: identity -> double -> add_one
    // Stage 0 (identity): x stays, y stays
    // Stage 1 (double): x*2, y*2
    // Stage 2 (add_one): x*2+1, y*2+1
    // So: [3,21, 5,41, 7,61, 9,81, 11,101, 13,121, 15,141, 17,161]
    let expected: Vec<f32> = vec![
        3.0, 21.0, // (1*2+1, 10*2+1)
        5.0, 41.0, // (2*2+1, 20*2+1)
        7.0, 61.0, // (3*2+1, 30*2+1)
        9.0, 81.0, // (4*2+1, 40*2+1)
        11.0, 101.0, // (5*2+1, 50*2+1)
        13.0, 121.0, // (6*2+1, 60*2+1)
        15.0, 141.0, // (7*2+1, 70*2+1)
        17.0, 161.0, // (8*2+1, 80*2+1)
    ];

    println!("Expected: {:?}", expected);

    // Verify output
    if output == expected {
        println!("✓ Output matches expected result!");
    } else {
        println!("✗ Output does not match!");
        for i in 0..output.len() {
            if (output[i] - expected[i]).abs() > 0.001 {
                println!("  Index {}: got {}, expected {}", i, output[i], expected[i]);
            }
        }
    }

    Ok(())
}
