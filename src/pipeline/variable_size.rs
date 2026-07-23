//! Variable-size buffer support for ping-pong pipeline.
//!
//! This module provides functionality for pipelines where each stage can have
//! independent buffer sizes, allowing for efficient memory usage when different
//! stages require different capacities.
//!
//! # Design
//!
//! Instead of requiring all stages to have the same batch_size and vector_dim,
//! this module allows each stage to specify its own requirements. When data
//! flows between stages with different sizes, the pipeline handles conversion:
//!
//! - **Growing** (smaller → larger): Data is copied with zero-padding
//! - **Shrinking** (larger → smaller): Only the first N elements are copied
//!
//! # Usage
//!
//! ```ignore
//! use wgsl_ping_pong_pipeline::pipeline::{Pipeline, Stage, VariableSizePipeline};
//!
//! // Create a pipeline with different-sized stages
//! let stage1 = Stage::new("stage1", shader1, 2, 1024);  // 1024 elements
//! let stage2 = Stage::new("stage2", shader2, 2, 2048);  // 2048 elements
//!
//! let pipeline = VariableSizePipeline::builder()
//!     .pipe(stage1)
//!     .pipe(stage2)
//!     .build()
//!     .await?;
//!
//! // Resize only stage2 to 4096
//! pipeline.resize_stage(1, 4096).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{bail, Result};
use wgpu::CommandEncoder;

use crate::wgpu_utils::{ComputeContext, stage_buffer_usages, readback_buffer_usages};
use super::{PipelineStage, StageConfig, Stage, F32_SIZE};

/// Per-stage configuration with output buffer size.
/// 
/// Each stage specifies its output buffer size. The input buffer size
/// is automatically determined by the previous stage's output buffer size.
#[derive(Debug, Clone)]
pub struct StageSizeConfig<T> {
    /// The stage configuration (standard or custom)
    pub config: StageConfig<T>,
    /// Output buffer size for this stage (in elements)
    /// This determines the size of this stage's output buffers.
    /// For stage 0, this also determines the input buffer size (external input).
    pub output_batch_size: usize,
    /// Vector dimension for this stage
    pub vector_dim: usize,
}

impl<T> StageSizeConfig<T> {
    pub fn new(config: StageConfig<T>, output_batch_size: usize, vector_dim: usize) -> Self {
        Self {
            config,
            output_batch_size,
            vector_dim,
        }
    }

    pub fn name(&self) -> &str {
        self.config.name()
    }

    pub fn element_size(&self) -> usize {
        self.vector_dim * F32_SIZE
    }

    /// Buffer size in bytes for this stage's output
    pub fn buffer_size(&self) -> u64 {
        self.output_batch_size as u64 * self.element_size() as u64
    }
}

/// A pipeline that supports variable buffer sizes per stage.
///
/// Unlike the standard Pipeline which requires all stages to have the same
/// batch_size and vector_dim, this pipeline allows each stage to have
/// independent sizing requirements.
///
/// # Buffer Organization
///
/// For N stages with variable sizes:
/// - Buffers 0-1: Stage 0 input buffers (size = stage0.input_batch_size)
/// - Buffers 2-3: Stage 0 output buffers (size = stage0.output_batch_size)
/// - Buffers 4-5: Stage 1 output buffers (size = stage1.output_batch_size)
/// - ...
///
/// Note: Stage i input is aliased from Stage i-1 output buffers, so we don't
/// duplicate input buffers for each stage.
#[derive(Debug)]
pub struct VariableSizePipeline<T> {
    context: Arc<ComputeContext>,
    /// Current tick count
    tick_count: u64,
    /// Stage configurations with their size requirements
    stage_configs: Vec<StageSizeConfig<T>>,
    /// All buffers: [stage0_input_a, stage0_input_b, stage0_out_a, stage0_out_b, stage1_out_a, stage1_out_b, ...]
    /// Stage i input buffers are at indices 0-1 (shared) or aliased from previous stage output
    /// Stage i output buffers are at indices 2 + 2*i and 2 + 2*i + 1
    buffers: Vec<Arc<wgpu::Buffer>>,
    /// Compute pipelines for standard stages
    compute_pipelines: Vec<Option<Arc<wgpu::ComputePipeline>>>,
    /// Bind group layouts for each stage
    bind_group_layouts: Vec<Arc<wgpu::BindGroupLayout>>,
    /// Bind groups for standard stages: [[input_variant][output_state]]
    bind_groups: Vec<Option<[[wgpu::BindGroup; 2]; 2]>>,
    /// Current output buffer index for each stage (0 or 1)
    current_output_indices: Vec<usize>,
    /// Current input buffer index to write to (0 or 1)
    current_input_write_index: usize,
    /// Side input buffers
    side_inputs: HashMap<String, Arc<wgpu::Buffer>>,
    /// Last output tag
    last_output_tag: Option<T>,
    /// Buffer metadata for tracking submission sizes
    buffer_submission_metadata: Vec<Option<(usize, usize, usize)>>,
    /// Default n value
    default_n: usize,
}

impl<T: Clone> VariableSizePipeline<T> {
    /// Creates a new variable-size pipeline builder
    pub fn builder() -> VariableSizePipelineBuilder<T> {
        VariableSizePipelineBuilder::new()
    }

    /// Gets the number of stages
    pub fn num_stages(&self) -> usize {
        self.stage_configs.len()
    }

    /// Gets the output buffer size for a specific stage
    pub fn stage_output_buffer_size(&self, stage_idx: usize) -> u64 {
        self.stage_configs[stage_idx].buffer_size()
    }

    /// Gets the input buffer size (stage 0's buffer size)
    pub fn input_buffer_size(&self) -> u64 {
        self.stage_configs[0].buffer_size()
    }

    /// Resizes a specific stage's output buffers to a new size.
    /// 
    /// In a ping-pong buffer system, only ONE buffer from each pair should be resized
    /// at a time. The other buffer retains its old size until the next tick when the
    /// roles swap.
    /// 
    /// This method resizes the buffer that is NOT currently being written to (the
    /// "read" buffer in the ping-pong pair), ensuring that in-flight data is not corrupted.
    /// 
    /// # Arguments
    /// * `stage_idx` - The stage index (0 for input buffers, 1+ for stage output buffers)
    /// * `new_batch_size` - The new batch size for this stage's buffers
    pub async fn resize_stage(&mut self, stage_idx: usize, new_batch_size: usize) -> Result<()> {
        if stage_idx > self.stage_configs.len() {
            bail!("Stage index {} out of bounds (max {})", stage_idx, self.stage_configs.len());
        }

        // Determine which buffer to resize (only ONE from the ping-pong pair)
        // We resize the buffer that is NOT currently being written to
        let buf_idx = if stage_idx == 0 {
            // Input buffers: resize the one NOT being written to
            // current_input_write_index points to the next buffer to write to
            // So the other buffer (1 - current_input_write_index) is safe to resize
            self.stage_configs[0].output_batch_size = new_batch_size;
            1 - self.current_input_write_index
        } else {
            // Output buffers for stage (stage_idx - 1)
            let stage_config = &mut self.stage_configs[stage_idx - 1];
            stage_config.output_batch_size = new_batch_size;
            let base_idx = 2 + 2 * (stage_idx - 1);
            // Resize the buffer that is NOT the current output target
            // current_output_indices[stage_idx-1] is the current write target
            // So resize the other one: 1 - current_output_indices[stage_idx-1]
            base_idx + (1 - self.current_output_indices[stage_idx - 1])
        };

        let element_size = self.stage_configs[0].element_size(); // All stages have same vector_dim
        let new_buffer_size = new_batch_size as u64 * element_size as u64;
        let usages = stage_buffer_usages();

        // Create new buffer and copy existing data
        let old_buffer = &self.buffers[buf_idx];
        let old_size = old_buffer.size();
        let copy_size = std::cmp::min(old_size, new_buffer_size);

        let new_buffer = Arc::new(self.context.create_buffer(
            Some(&format!("Buffer {} Resized to {}", buf_idx, new_batch_size)),
            new_buffer_size,
            usages,
        ));

        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Stage {} Resize Encoder", stage_idx)),
            },
        );

        // Copy existing data to new buffer
        if copy_size > 0 {
            encoder.copy_buffer_to_buffer(old_buffer, 0, &new_buffer, 0, copy_size);
        }

        self.buffers[buf_idx] = new_buffer;
        
        // Clear metadata for this buffer since it's been resized
        if buf_idx < self.buffer_submission_metadata.len() {
            self.buffer_submission_metadata[buf_idx] = None;
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.context.device_poll()?;

        // Clear bind groups for affected stages since buffer sizes changed
        let affected_stage = if stage_idx == 0 { 0 } else { stage_idx - 1 };
        for i in affected_stage..self.bind_groups.len() {
            self.bind_groups[i] = None;
        }

        Ok(())
    }

    /// Resizes both buffers in a stage's ping-pong pair to the same size.
    /// 
    /// This is useful when you want to ensure both buffers have the same size,
    /// but it may cause issues if there's data in-flight. Use with caution.
    /// 
    /// Prefer `resize_stage()` which only resizes one buffer at a time for
    /// proper ping-pong behavior.
    /// 
    /// # Arguments
    /// * `stage_idx` - The stage index (0 for input buffers, 1+ for stage output buffers)
    /// * `new_batch_size` - The new batch size for both buffers
    pub async fn resize_stage_both(&mut self, stage_idx: usize, new_batch_size: usize) -> Result<()> {
        if stage_idx > self.stage_configs.len() {
            bail!("Stage index {} out of bounds (max {})", stage_idx, self.stage_configs.len());
        }

        // Determine which buffers to resize (both from the ping-pong pair)
        let buffer_indices: Vec<usize> = if stage_idx == 0 {
            self.stage_configs[0].output_batch_size = new_batch_size;
            vec![0, 1]
        } else {
            let stage_config = &mut self.stage_configs[stage_idx - 1];
            stage_config.output_batch_size = new_batch_size;
            let base_idx = 2 + 2 * (stage_idx - 1);
            vec![base_idx, base_idx + 1]
        };

        let element_size = self.stage_configs[0].element_size();
        let new_buffer_size = new_batch_size as u64 * element_size as u64;
        let usages = stage_buffer_usages();

        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Stage {} Resize Both Encoder", stage_idx)),
            },
        );

        for &buf_idx in &buffer_indices {
            let old_buffer = &self.buffers[buf_idx];
            let old_size = old_buffer.size();
            let copy_size = std::cmp::min(old_size, new_buffer_size);

            let new_buffer = Arc::new(self.context.create_buffer(
                Some(&format!("Buffer {} Resized to {}", buf_idx, new_batch_size)),
                new_buffer_size,
                usages,
            ));

            if copy_size > 0 {
                encoder.copy_buffer_to_buffer(old_buffer, 0, &new_buffer, 0, copy_size);
            }

            self.buffers[buf_idx] = new_buffer;
            if buf_idx < self.buffer_submission_metadata.len() {
                self.buffer_submission_metadata[buf_idx] = None;
            }
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.context.device_poll()?;

        let affected_stage = if stage_idx == 0 { 0 } else { stage_idx - 1 };
        for i in affected_stage..self.bind_groups.len() {
            self.bind_groups[i] = None;
        }

        Ok(())
    }

    /// Gets the current buffer sizes for a stage's output ping-pong pair
    /// Returns (buffer_a_size, buffer_b_size) in elements
    /// 
    /// # Arguments
    /// * `stage_idx` - The stage index (0 for input buffers, 1+ for output buffers)
    pub fn get_stage_buffer_sizes(&self, stage_idx: usize) -> (usize, usize) {
        let element_size = self.stage_configs[0].element_size() as u64;
        
        if stage_idx == 0 {
            // Input buffers are at indices 0 and 1
            let buf_a_size = self.buffers[0].size() / element_size;
            let buf_b_size = self.buffers[1].size() / element_size;
            (buf_a_size as usize, buf_b_size as usize)
        } else {
            // Stage i output buffers are at indices 2 + 2*(i-1) and 2 + 2*(i-1) + 1
            // Note: stage_idx 1 = stage 0's output, stage_idx 2 = stage 1's output, etc.
            // But we need to be careful: for N stages, we have N+1 buffer pairs:
            // - Pair 0: input buffers (2 buffers)
            // - Pair 1: stage 0 output buffers (2 buffers)
            // - Pair 2: stage 1 output buffers (2 buffers)
            // So for stage_idx=2, we want pair 2 = stage 1 output = buffers 4 and 5
            let pair_idx = stage_idx; // stage_idx 0 = input pair, 1 = stage 0 output pair, 2 = stage 1 output pair
            let base_idx = 2 * pair_idx; // pair 0: 0-1, pair 1: 2-3, pair 2: 4-5
            let buf_a_size = self.buffers[base_idx].size() / element_size;
            let buf_b_size = self.buffers[base_idx + 1].size() / element_size;
            (buf_a_size as usize, buf_b_size as usize)
        }
    }

    /// Writes input data for the current write buffer
    pub async fn write_input<D: bytemuck::Pod>(&mut self, data: &[D]) -> Result<()> {
        let stage_0_config = &self.stage_configs[0];
        let expected_byte_size = stage_0_config.buffer_size() as usize;
        let actual_byte_size = data.len() * std::mem::size_of::<D>();

        if actual_byte_size > expected_byte_size {
            bail!(
                "Input size {} bytes exceeds stage 0 buffer size {} bytes",
                actual_byte_size, expected_byte_size
            );
        }

        let write_idx = self.current_input_write_index;
        self.context.queue.write_buffer(&self.buffers[write_idx], 0, bytemuck::cast_slice(data));
        self.current_input_write_index = 1 - write_idx;

        Ok(())
    }

    /// Process one tick through the pipeline
    pub async fn tick(&mut self, tag: T) -> Result<()> {
        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Tick {} Encoder", self.tick_count)),
            },
        );

        // Process tag flow
        let mut current_tag: Option<T> = Some(tag.clone());
        for stage_config in &mut self.stage_configs {
            current_tag = stage_config.config.forward_tag(current_tag);
        }

        // In immediate mode, output is ready after each tick
        self.last_output_tag = Some(tag);

        // Process each stage
        for i in 0..self.stage_configs.len() {
            let state = self.current_output_indices[i];

            // Calculate input buffer index
            // In immediate mode, all stages process in a single tick
            // Stage 0 reads from the input buffer, subsequent stages read from previous stage's current output
            let input_buffer_idx = if i == 0 {
                1 - self.current_input_write_index
            } else {
                // Read from the buffer that the previous stage is writing to in THIS tick
                2 + 2 * (i - 1) + self.current_output_indices[i - 1]
            };

            let output_buffer_idx = 2 + 2 * i + state;

            // Get submission metadata
            let metadata = self.buffer_submission_metadata[input_buffer_idx];
            let (actual_elements, n) = match metadata {
                Some((actual_elements, n, _batch_size)) => (actual_elements, n),
                None => (self.stage_configs[i].output_batch_size, self.default_n),
            };

            // Update stage with metadata
            self.stage_configs[i].config.update_actual_total_elements(actual_elements)?;
            self.stage_configs[i].config.update_n(n)?;

            // Process the stage
            match &self.stage_configs[i].config {
                StageConfig::Standard { .. } => {
                    if let Some(compute_pipeline) = &self.compute_pipelines[i] {
                        // Lazy bind group initialization
                        if self.bind_groups[i].is_none() {
                            let bgs = [
                                [self.create_bind_group_for_stage(i, 0, 0), self.create_bind_group_for_stage(i, 0, 1)],
                                [self.create_bind_group_for_stage(i, 1, 0), self.create_bind_group_for_stage(i, 1, 1)],
                            ];
                            self.bind_groups[i] = Some(bgs);
                        }

                        let (input_buffer_variant, output_state) = if i == 0 {
                            (1 - self.current_input_write_index, state)
                        } else {
                            (1 - state, state)
                        };

                        let bind_group = &self.bind_groups[i].as_ref().unwrap()[input_buffer_variant][output_state];

                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!("Stage {} Pass Tick {}", self.stage_configs[i].name(), self.tick_count)),
                            timestamp_writes: None,
                        });

                        pass.set_pipeline(compute_pipeline);
                        pass.set_bind_group(0, bind_group, &[]);

                        let workgroup_size = 64u32;
                        let dispatch_count = (self.stage_configs[i].output_batch_size as u32 + workgroup_size - 1) / workgroup_size;
                        pass.dispatch_workgroups(dispatch_count, 1, 1);
                    }
                }
                StageConfig::Custom { stage, .. } => {
                    let input_buffer = &self.buffers[input_buffer_idx];
                    let output_buffer = &self.buffers[output_buffer_idx];
                    stage.encode(&mut encoder, input_buffer, output_buffer, &self.side_inputs)?;
                }
            }

            // Propagate metadata
            self.buffer_submission_metadata[output_buffer_idx] = self.buffer_submission_metadata[input_buffer_idx];
        }

        // Flip output indices
        for i in 0..self.current_output_indices.len() {
            self.current_output_indices[i] = 1 - self.current_output_indices[i];
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.context.device_poll()?;
        self.tick_count += 1;

        Ok(())
    }

    /// Creates a bind group for a specific stage
    fn create_bind_group_for_stage(&self, stage_idx: usize, input_buffer_variant: usize, output_state: usize) -> wgpu::BindGroup {
        let bgl = &self.bind_group_layouts[stage_idx];
        let input_buffer_idx = if stage_idx == 0 {
            input_buffer_variant
        } else {
            2 + 2 * (stage_idx - 1) + input_buffer_variant
        };
        let output_buffer_idx = 2 + 2 * stage_idx + output_state;

        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.buffers[input_buffer_idx].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.buffers[output_buffer_idx].as_entire_binding(),
            },
        ];

        self.context.create_bind_group(
            Some(&format!("Stage {} BG Input {} Output {}", self.stage_configs[stage_idx].name(), input_buffer_variant, output_state)),
            bgl,
            &entries,
        )
    }

    /// Reads the output from the last stage
    pub async fn read_output(&self) -> Result<Option<(Option<T>, Vec<f32>)>> {
        if self.last_output_tag.is_none() {
            return Ok(None);
        }

        let num_stages = self.compute_pipelines.len();
        if num_stages == 0 {
            bail!("Pipeline has no stages");
        }

        // Output is in the last stage's current output buffer
        let last_stage_idx = num_stages - 1;
        let last_output_buffer_idx = 2 + 2 * last_stage_idx + (1 - self.current_output_indices[last_stage_idx]);
        let read_buffer = &self.buffers[last_output_buffer_idx];

        // Use actual buffer size, not config size, since buffers may have been selectively resized
        let buffer_size = read_buffer.size();
        let element_count = buffer_size as usize / F32_SIZE;

        let readback_buffer = self.context.create_buffer(
            Some("Output Readback"),
            buffer_size,
            readback_buffer_usages(),
        );

        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            },
        );

        encoder.copy_buffer_to_buffer(read_buffer, 0, &readback_buffer, 0, buffer_size);
        self.context.queue.submit(Some(encoder.finish()));
        self.context.device_poll()?;

        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        use wgpu::PollType;
        self.context.device.poll(PollType::Wait { submission_index: None, timeout: None })?;

        let _result = receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("Channel closed"))??;
        let data: &[u8] = &buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data[..element_count * F32_SIZE]).to_vec();

        Ok(Some((self.last_output_tag.clone(), result)))
    }
}

/// Builder for variable-size pipelines
#[derive(Debug, Default)]
pub struct VariableSizePipelineBuilder<T> {
    stage_configs: Vec<StageSizeConfig<T>>,
    context: Option<Arc<ComputeContext>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Clone> VariableSizePipelineBuilder<T> {
    pub fn new() -> Self {
        Self {
            stage_configs: Vec::new(),
            context: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a standard stage with a specific output buffer size
    pub fn pipe_with_size(mut self, stage: Stage, output_batch_size: usize) -> Self {
        let vector_dim = stage.vector_dim;
        let config = StageSizeConfig {
            config: StageConfig::Standard { stage, tag: None },
            output_batch_size,
            vector_dim,
        };
        self.stage_configs.push(config);
        self
    }

    /// Add a custom stage with a specific output buffer size
    pub fn pipe_custom_with_size(mut self, stage: Box<dyn PipelineStage>, output_batch_size: usize) -> Self {
        let vector_dim = stage.vector_dim();
        let config = StageSizeConfig {
            config: StageConfig::Custom { stage, tag: None },
            output_batch_size,
            vector_dim,
        };
        self.stage_configs.push(config);
        self
    }

    /// Set a custom ComputeContext
    pub fn with_context(mut self, context: Arc<ComputeContext>) -> Self {
        self.context = Some(context);
        self
    }

    /// Build the variable-size pipeline
    pub async fn build(mut self) -> Result<VariableSizePipeline<T>> {
        if self.stage_configs.is_empty() {
            bail!("Pipeline must have at least one stage");
        }

        // Verify vector_dim consistency (still required for data compatibility)
        let vector_dim = self.stage_configs[0].vector_dim;
        for config in &self.stage_configs {
            if config.vector_dim != vector_dim {
                bail!(
                    "All stages must have the same vector dimension. Stage '{}' has {}, expected {}",
                    config.name(), config.vector_dim, vector_dim
                );
            }
        }

        let context = match self.context {
            Some(ctx) => ctx,
            None => Arc::new(ComputeContext::new_high_performance().await?),
        };

        let default_bind_group_layout = Arc::new(create_bind_group_layout(&context));

        // Create all buffers
        let mut buffers = Vec::new();

        // Input buffers (2 for ping-pong) - size determined by stage 0's output size (which is also its input size)
        let input_size = self.stage_configs[0].buffer_size();
        buffers.push(Arc::new(context.create_buffer(
            Some("Variable Pipeline Input Buffer A"),
            input_size,
            stage_buffer_usages(),
        )));
        buffers.push(Arc::new(context.create_buffer(
            Some("Variable Pipeline Input Buffer B"),
            input_size,
            stage_buffer_usages(),
        )));

        // Output buffers for each stage - each stage has its own output size
        for stage_config in &self.stage_configs {
            let buffer_size = stage_config.buffer_size();
            buffers.push(Arc::new(context.create_buffer(
                Some(&format!("Stage {} Output Buffer A", stage_config.name())),
                buffer_size,
                stage_buffer_usages(),
            )));
            buffers.push(Arc::new(context.create_buffer(
                Some(&format!("Stage {} Output Buffer B", stage_config.name())),
                buffer_size,
                stage_buffer_usages(),
            )));
        }

        // Create compute pipelines and bind group layouts
        let mut compute_pipelines = Vec::with_capacity(self.stage_configs.len());
        let mut bind_group_layouts = Vec::with_capacity(self.stage_configs.len());
        let mut stage_names = Vec::with_capacity(self.stage_configs.len());

        // Initialize custom stages
        for stage_config in &mut self.stage_configs {
            if let StageConfig::Custom { stage, .. } = &mut stage_config.config {
                if stage.requires_initialization() {
                    stage.initialize(&context)?;
                }
            }
        }

        for stage_config in &self.stage_configs {
            let name = stage_config.name().to_string();
            let bgl = Arc::clone(&default_bind_group_layout);

            let pipeline = match &stage_config.config {
                StageConfig::Standard { stage, .. } => {
                    Some(Arc::new(context.create_compute_pipeline(
                        Some(&format!("Stage {} Pipeline", name)),
                        &stage.wgsl,
                        &[&*bgl],
                    )?))
                }
                StageConfig::Custom { .. } => {
                    None
                }
            };

            compute_pipelines.push(pipeline);
            bind_group_layouts.push(bgl);
            stage_names.push(name);
        }

        let num_buffers = buffers.len();
        let num_stages = self.stage_configs.len();
        let default_n = self.stage_configs[0].output_batch_size;

        Ok(VariableSizePipeline {
            context,

            tick_count: 0,
            stage_configs: self.stage_configs,
            buffers,
            compute_pipelines,
            bind_group_layouts,
            bind_groups: vec![None; num_stages],
            current_output_indices: vec![0; num_stages],
            current_input_write_index: 0,
            side_inputs: HashMap::new(),
            last_output_tag: None,
            buffer_submission_metadata: vec![None; num_buffers],
            default_n,
        })
    }
}

fn create_bind_group_layout(context: &ComputeContext) -> wgpu::BindGroupLayout {
    context.create_bind_group_layout(
        Some("Variable Stage Bind Group Layout"),
        &[
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
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pollster::block_on;

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

    /// Simple custom stage for testing
    #[derive(Debug)]
    struct TestCustomStage {
        name: String,
        vector_dim: usize,
        batch_size: usize,
    }

    impl PipelineStage for TestCustomStage {
        fn name(&self) -> &str { &self.name }
        fn vector_dim(&self) -> usize { self.vector_dim }
        fn batch_size(&self) -> usize { self.batch_size }
        fn encode(&self, _encoder: &mut CommandEncoder, _input: &wgpu::Buffer, _output: &wgpu::Buffer, _side_inputs: &HashMap<String, Arc<wgpu::Buffer>>) -> Result<()> {
            Ok(())
        }
        fn initialize(&mut self, _context: &ComputeContext) -> Result<()> { Ok(()) }
        fn requires_initialization(&self) -> bool { false }
    }

    #[test]
    fn test_builder_creation() {
        let stage = Stage::new("test", DOUBLE_WGSL, 1, 1024);
        let builder = VariableSizePipelineBuilder::<u64>::new()
            .pipe_with_size(stage, 2048);
        assert_eq!(builder.stage_configs.len(), 1);
        assert_eq!(builder.stage_configs[0].output_batch_size, 2048);
    }

    #[test]
    fn test_per_stage_buffer_sizes() {
        let stage1 = Stage::new("stage1", DOUBLE_WGSL, 1, 1024);
        let stage2 = Stage::new("stage2", DOUBLE_WGSL, 1, 2048);
        let stage3 = Stage::new("stage3", DOUBLE_WGSL, 1, 4096);

        let builder = VariableSizePipelineBuilder::<u64>::new()
            .pipe_with_size(stage1, 1024)
            .pipe_with_size(stage2, 2048)
            .pipe_with_size(stage3, 4096);

        assert_eq!(builder.stage_configs[0].output_batch_size, 1024);
        assert_eq!(builder.stage_configs[1].output_batch_size, 2048);
        assert_eq!(builder.stage_configs[2].output_batch_size, 4096);
    }

    #[test]
    fn test_selective_stage_resize() {
        let stage1 = Stage::new("stage1", DOUBLE_WGSL, 1, 1024);
        let stage2 = Stage::new("stage2", DOUBLE_WGSL, 1, 1024);
        let stage3 = Stage::new("stage3", DOUBLE_WGSL, 1, 1024);

        let builder = VariableSizePipelineBuilder::<u64>::new()
            .pipe_with_size(stage1, 1024)
            .pipe_with_size(stage2, 1024)
            .pipe_with_size(stage3, 1024);

        // Initially all stages have output_batch_size = 1024
        assert_eq!(builder.stage_configs[0].output_batch_size, 1024);
        assert_eq!(builder.stage_configs[1].output_batch_size, 1024);
        assert_eq!(builder.stage_configs[2].output_batch_size, 1024);
        
        // We can't test the actual resize without running async code,
        // but we can verify the builder creates correct configs
    }
}
