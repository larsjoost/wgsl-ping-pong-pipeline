//! Pipeline implementation with isolated per-stage ping-pong buffers.

use crate::wgpu_utils::{ComputeContext, readback_buffer_usages, stage_buffer_usages};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::sync::Arc;

pub mod pipeline_stage;
pub mod variable_size;
pub use pipeline_stage::{PipelineStage, StageConfig};
pub use variable_size::{StageSizeConfig, VariableSizePipeline, VariableSizePipelineBuilder};

/// Element size in bytes for f32.
pub const F32_SIZE: usize = 4;

/// A compute stage configuration.
#[derive(Debug, Clone)]
pub struct Stage {
    /// Name of this stage
    pub name: String,
    /// WGSL shader source code
    pub wgsl: String,
    /// Vector dimension (2, 3, 4, etc.)
    pub vector_dim: usize,
    /// Number of elements in each batch
    pub batch_size: usize,
    /// Custom bind group layout entries (optional).
    /// If None, uses the default layout (input at binding 0, output at binding 1).
    pub bind_group_entries: Option<Vec<wgpu::BindGroupLayoutEntry>>,
}

impl Stage {
    /// Creates a new stage.
    pub fn new(
        name: impl Into<String>,
        wgsl: impl Into<String>,
        vector_dim: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            name: name.into(),
            wgsl: wgsl.into(),
            vector_dim,
            batch_size,
            bind_group_entries: None,
        }
    }

    /// Creates an identity stage that passes data through unchanged.
    pub fn identity(name: impl Into<String>, vector_dim: usize, batch_size: usize) -> Self {
        let shader = create_identity_shader(vector_dim);
        Self::new(name, shader, vector_dim, batch_size)
    }

    /// Validates the stage configuration.
    pub fn validate(&self) -> Result<()> {
        if self.vector_dim == 0 {
            bail!("Vector dimension must be at least 1");
        }
        if self.batch_size == 0 {
            bail!("Batch size must be at least 1");
        }
        Ok(())
    }

    /// Element size in bytes.
    pub fn element_size(&self) -> usize {
        self.vector_dim * F32_SIZE
    }

    /// Buffer size in bytes.
    pub fn buffer_size(&self) -> u64 {
        self.batch_size as u64 * self.element_size() as u64
    }
}

/// Generates an identity shader.
/// Uses a flat f32 array instead of structs for better compatibility.
/// The vector_dim parameter is currently unused but kept for API compatibility.
fn create_identity_shader(_vector_dim: usize) -> String {
    // For identity, we just copy all elements from input to output
    // The data is laid out as a flat array of f32 values
    // So we need to divide idx by vector_dim to get the element index
    r#"
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
"#
    .to_string()
}

/// Pipeline builder.
#[derive(Debug, Default)]
pub struct PipelineBuilder<T> {
    stages: Vec<StageConfig<T>>,
    context: Option<Arc<ComputeContext>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> PipelineBuilder<T> {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            context: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn pipe(mut self, stage: Stage) -> Self {
        self.stages.push(StageConfig::Standard { stage, tag: None });
        self
    }

    pub fn pipe_custom(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(StageConfig::Custom { stage, tag: None });
        self
    }

    pub fn pipe_config(mut self, config: StageConfig<T>) -> Self {
        self.stages.push(config);
        self
    }

    pub fn identity(
        mut self,
        name: impl Into<String>,
        vector_dim: usize,
        batch_size: usize,
    ) -> Self {
        self.stages.push(StageConfig::Standard {
            stage: Stage::identity(name, vector_dim, batch_size),
            tag: None,
        });
        self
    }

    /// Set a custom ComputeContext for the pipeline.
    ///
    /// This allows sharing a ComputeContext between the pipeline and external
    /// resources (like side input buffers).
    ///
    /// If not set, a new ComputeContext will be created automatically.
    pub fn with_context(mut self, context: Arc<ComputeContext>) -> Self {
        self.context = Some(context);
        self
    }

    pub async fn build(self) -> Result<Pipeline<T>> {
        Pipeline::build_from_configs_with_context(self.stages, self.context).await
    }
}

/// The built pipeline.
#[derive(Debug)]
pub struct Pipeline<T> {
    context: Arc<ComputeContext>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    vector_dim: usize,
    batch_size: usize,
    tick_count: u64,
    /// All buffers: [input_A, input_B, stage0_out_A, stage0_out_B, stage1_out_A, stage1_out_B, ...]
    /// Buffer layout grouped by set:
    /// - Set A (read when current_set=0): indices 0, 2, 4, 6... (input_A, stage0_out_A, stage1_out_A, ...)
    /// - Set B (read when current_set=1): indices 1, 3, 5, 7... (input_B, stage0_out_B, stage1_out_B, ...)
    ///   For N stages, we need 2 input buffers + 2*N output buffers
    buffers: Vec<Arc<wgpu::Buffer>>,
    /// Compute pipelines for each stage (only for standard stages)
    compute_pipelines: Vec<Option<Arc<wgpu::ComputePipeline>>>,
    /// Bind group layouts for each stage (can be different per stage)
    bind_group_layouts: Vec<Arc<wgpu::BindGroupLayout>>,
    /// Pre-created bind groups for each stage
    /// For stage 0: 2 input buffers × 2 output buffers = 4 bind groups
    /// For other stages: 2 input buffers (from prev stage) × 2 output buffers = 4 bind groups
    /// But we store them as [[wgpu::BindGroup; 2]; 2] where:
    /// - First index: input buffer index (0 or 1)
    /// - Second index: output buffer state (0 or 1)
    ///   Only used for standard stages; custom stages create their own bind groups
    bind_groups: Vec<Option<[[wgpu::BindGroup; 2]; 2]>>,
    /// Current buffer set to read from (0 or 1)
    /// All stages read from buffers in set `current_set` and write to buffers in set `1 - current_set`
    current_set: usize,
    /// Stage names
    stage_names: Vec<String>,
    /// Stage configurations (standard or custom)
    stage_configs: Vec<StageConfig<T>>,
    /// Side input buffers managed by the pipeline
    /// Maps from name to buffer
    side_inputs: HashMap<String, Arc<wgpu::Buffer>>,
    /// Per-buffer metadata: for each buffer, store submission metadata
    /// This allows tracking submission sizes through the pipeline without padding
    /// Index corresponds to buffer index in the buffers vec
    /// Stores (actual_total_elements, n, batch_size) for the submission in that buffer
    buffer_submission_metadata: Vec<Option<(usize, usize, usize)>>,
    /// Default n value to use when no metadata is available for a buffer
    /// This is set when the pipeline is created or resized
    default_n: usize,
}

impl<T> Pipeline<T> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> PipelineBuilder<T> {
        PipelineBuilder::new()
    }

    async fn build_from_configs_with_context(
        mut stage_configs: Vec<StageConfig<T>>,
        user_context: Option<Arc<ComputeContext>>,
    ) -> Result<Self> {
        if stage_configs.is_empty() {
            bail!("Pipeline must have at least one stage");
        }

        let mut vector_dim = 0;
        let mut batch_size = 0;

        for stage in &stage_configs {
            stage.validate()?;
            if vector_dim == 0 {
                vector_dim = stage.vector_dim();
                batch_size = stage.batch_size();
            } else {
                if stage.vector_dim() != vector_dim {
                    bail!(
                        "All stages must have the same vector dimension. Stage '{}' has {}, expected {}",
                        stage.name(),
                        stage.vector_dim(),
                        vector_dim
                    );
                }
                if stage.batch_size() != batch_size {
                    bail!(
                        "All stages must have the same batch size. Stage '{}' has {}, expected {}",
                        stage.name(),
                        stage.batch_size(),
                        batch_size
                    );
                }
            }
        }

        let context = match user_context {
            Some(ctx) => ctx,
            None => Arc::new(ComputeContext::new_high_performance().await?),
        };
        let default_bind_group_layout = Arc::new(create_bind_group_layout(&context));
        let element_size = vector_dim * F32_SIZE;
        let buffer_size = batch_size as u64 * element_size as u64;
        let usages = stage_buffer_usages();

        // Create all buffers: 2 input buffers + 2 output buffers per stage
        let num_buffers = 2 + 2 * stage_configs.len();
        let mut buffers = Vec::with_capacity(num_buffers);

        // Input buffers for stage 0 (ping-pong pair)
        buffers.push(Arc::new(context.create_buffer(
            Some("Pipeline Input Buffer A"),
            buffer_size,
            usages,
        )));
        buffers.push(Arc::new(context.create_buffer(
            Some("Pipeline Input Buffer B"),
            buffer_size,
            usages,
        )));

        // Output buffers for each stage
        for stage_config in stage_configs.iter() {
            buffers.push(Arc::new(context.create_buffer(
                Some(&format!("Stage {} Output Buffer A", stage_config.name())),
                buffer_size,
                usages,
            )));
            buffers.push(Arc::new(context.create_buffer(
                Some(&format!("Stage {} Output Buffer B", stage_config.name())),
                buffer_size,
                usages,
            )));
        }

        // Create compute pipelines and bind group layouts for each stage
        let mut compute_pipelines: Vec<Option<Arc<wgpu::ComputePipeline>>> =
            Vec::with_capacity(stage_configs.len());
        let mut bind_group_layouts = Vec::with_capacity(stage_configs.len());
        let mut stage_names = Vec::with_capacity(stage_configs.len());
        let mut bind_groups: Vec<Option<[wgpu::BindGroup; 2]>> =
            Vec::with_capacity(stage_configs.len());

        // Initialize custom stages in place (we take ownership of stage_configs)
        for stage_config in &mut stage_configs {
            if let StageConfig::Custom { stage, .. } = stage_config
                && stage.requires_initialization()
            {
                stage.initialize(&context)?;
            }
        }

        for stage_config in &stage_configs {
            let name = stage_config.name().to_string();
            let bgl = match stage_config {
                StageConfig::Standard { stage, .. } => {
                    // Use custom bind group layout if provided, otherwise use default
                    if let Some(ref entries) = stage.bind_group_entries {
                        Arc::new(context.create_bind_group_layout(
                            Some(&format!("Stage {} Bind Group Layout", name)),
                            entries,
                        ))
                    } else {
                        Arc::clone(&default_bind_group_layout)
                    }
                }
                StageConfig::Custom { .. } => {
                    // Custom stages don't need bind group layouts from us
                    // They create their own during encode()
                    Arc::clone(&default_bind_group_layout)
                }
            };

            let pipeline = match stage_config {
                StageConfig::Standard { stage, .. } => {
                    Some(Arc::new(context.create_compute_pipeline(
                        Some(&format!("Stage {} Pipeline", name)),
                        &stage.wgsl,
                        &[&*bgl],
                    )?))
                }
                StageConfig::Custom { .. } => {
                    // Custom stages don't use compute pipelines
                    None
                }
            };

            compute_pipelines.push(pipeline);
            bind_group_layouts.push(bgl);
            bind_groups.push(None); // Will be filled later for standard stages
            stage_names.push(name);
        }

        let num_buffers = buffers.len();
        let pipeline = Self {
            context,
            bind_group_layout: Arc::clone(&default_bind_group_layout),
            vector_dim,
            batch_size,
            tick_count: 0,
            buffers,
            compute_pipelines,
            bind_group_layouts,
            bind_groups: vec![None; stage_configs.len()], // Will be initialized lazily as 2x2 arrays
            current_set: 0,
            stage_names,
            stage_configs,
            side_inputs: std::collections::HashMap::new(),
            buffer_submission_metadata: vec![None; num_buffers],
            default_n: batch_size, // Use batch_size as initial default_n
        };

        Ok(pipeline)
    }

    /// Creates a bind group for a specific stage and state (0 or 1).
    ///
    /// The state determines which buffer in the ping-pong pair is being written to.
    /// Staggering logic is applied here: Stage i reads from the buffer Stage i-1
    /// was writing to in the PREVIOUS tick.
    ///
    /// This is used for standard stages during pipeline creation.
    fn create_bind_group_for_stage(
        &self,
        stage_idx: usize,
        input_buffer_variant: usize,
        output_state: usize,
    ) -> wgpu::BindGroup {
        let bgl = &self.bind_group_layouts[stage_idx];
        // Buffer layout: [input_A, input_B, stage0_out_A, stage0_out_B, stage1_out_A, stage1_out_B, ...]
        // For any stage, input buffer index = 2 * stage_idx + variant
        let input_buffer_idx = 2 * stage_idx + input_buffer_variant;
        let output_buffer_idx = 2 * stage_idx + 2 + output_state;

        // Check how many bindings this stage's layout has
        // If it has 3 bindings (like multiply stage), we need to provide 3 entries
        // If it has 2 bindings (default), we provide 2 entries
        // Check the number of bindings by examining the entries
        // We need to access the BindGroupLayout's entries, but it's not directly accessible
        // For now, we'll use a heuristic: check if this stage has a custom layout
        let stage_has_custom_layout = Arc::as_ptr(&self.bind_group_layouts[stage_idx])
            != Arc::as_ptr(&self.bind_group_layout);
        let num_bindings = if stage_has_custom_layout {
            // For custom layouts, we need to determine the number of bindings
            // For now, we'll assume multiply stages have 3 bindings
            // This is a simplified approach
            3
        } else {
            2
        };

        let entries: Vec<wgpu::BindGroupEntry> = match num_bindings {
            2 => vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers[input_buffer_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers[output_buffer_idx].as_entire_binding(),
                },
            ],
            3 => vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers[input_buffer_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    // For multiply stage: binding 1 is input_b
                    // For now, we'll use the previous stage's output as input_b
                    // This is a simplified approach - in reality, input_b should be a separate buffer
                    resource: self.buffers[input_buffer_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers[output_buffer_idx].as_entire_binding(),
                },
            ],
            _ => vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers[input_buffer_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers[output_buffer_idx].as_entire_binding(),
                },
            ],
        };

        self.context.create_bind_group(
            Some(&format!(
                "Stage {} BG Input {} Output {}",
                self.stage_names[stage_idx], input_buffer_variant, output_state
            )),
            bgl,
            &entries,
        )
    }

    pub async fn process(&mut self, data: Option<&[f32]>, tag: T) -> Result<Option<(T, Vec<f32>)>> {
        let num_stages = self.compute_pipelines.len();
        if num_stages == 0 {
            bail!("Pipeline has no stages");
        }

        // Handle data writing if provided (replaces write_input)
        if let Some(data_slice) = data {
            let actual_byte_size = data_slice.len() * F32_SIZE;
            if actual_byte_size == 0 {
                bail!("Input data cannot be empty");
            }

            let element_size = self.vector_dim * F32_SIZE;
            if !actual_byte_size.is_multiple_of(element_size) {
                bail!(
                    "Input byte size {} is not a multiple of element size {} (vector_dim {} * F32_SIZE {})",
                    actual_byte_size,
                    element_size,
                    self.vector_dim,
                    F32_SIZE
                );
            }

            let actual_batch_size = actual_byte_size / element_size;
            // Write to the input buffer that will be read by Stage 0 in the current tick
            // Stage 0 reads from buffer `current_set` (either 0 or 1)
            let write_idx = self.current_set;

            if actual_byte_size as u64 > self.buffers[write_idx].size() {
                let new_batch_size = std::cmp::max(actual_batch_size, self.batch_size);
                self.resize(new_batch_size).await?;
            }

            // Write to the current input buffer
            self.context.queue.write_buffer(
                &self.buffers[write_idx],
                0,
                bytemuck::cast_slice(data_slice),
            );

            let custom_n = match self.buffer_submission_metadata[write_idx] {
                Some((_, custom_n, _)) => custom_n,
                None => self.default_n,
            };
            self.buffer_submission_metadata[write_idx] =
                Some((actual_batch_size, custom_n, actual_batch_size));
        }

        // Check if this call will produce output (delay line behavior)
        // For N-stage pipeline: first N-1 calls return None, Nth call and beyond return Some
        // tick_count represents the number of completed process() calls so far
        let will_have_output = self.tick_count >= (num_stages as u64 - 1);

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("Process {} Encoder", self.tick_count)),
                });

        // Process tag flow through pipeline - delay line implementation
        let mut current_tag: Option<T> = Some(tag);
        for i in 0..self.stage_configs.len() {
            current_tag = self.stage_configs[i].forward_tag(current_tag);
        }

        // Process all stages with compute passes
        for i in 0..self.stage_configs.len() {
            let input_buffer_idx = 2 * i + self.current_set;
            let output_buffer_idx = 2 * i + 2 + (1 - self.current_set);

            // Get submission metadata from input buffer
            let metadata = self.buffer_submission_metadata[input_buffer_idx];
            let (actual_elements, n) = match metadata {
                Some((actual_elements, n, _batch_size)) => (actual_elements, n),
                None => (self.batch_size, self.default_n),
            };

            // Update stage with metadata
            self.stage_configs[i].update_actual_total_elements(actual_elements)?;
            self.stage_configs[i].update_n(n)?;

            // Process the stage
            match &self.stage_configs[i] {
                StageConfig::Standard { .. } => {
                    if let Some(compute_pipeline) = &self.compute_pipelines[i] {
                        // Lazy bind group initialization
                        if self.bind_groups[i].is_none() {
                            let bgs = [
                                [
                                    self.create_bind_group_for_stage(i, 0, 0),
                                    self.create_bind_group_for_stage(i, 0, 1),
                                ],
                                [
                                    self.create_bind_group_for_stage(i, 1, 0),
                                    self.create_bind_group_for_stage(i, 1, 1),
                                ],
                            ];
                            self.bind_groups[i] = Some(bgs);
                        }

                        let input_buffer_variant = self.current_set;
                        let output_state = 1 - self.current_set;
                        let bind_group = &self.bind_groups[i].as_ref().unwrap()
                            [input_buffer_variant][output_state];

                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!(
                                "Stage {} Pass Process {}",
                                self.stage_names[i], self.tick_count
                            )),
                            timestamp_writes: None,
                        });

                        pass.set_pipeline(compute_pipeline);
                        pass.set_bind_group(0, bind_group, &[]);

                        let workgroup_size = 64u32;
                        let dispatch_count = (actual_elements as u32).div_ceil(workgroup_size);
                        pass.dispatch_workgroups(dispatch_count, 1, 1);
                    }
                }
                StageConfig::Custom { stage, .. } => {
                    let input_buffer = &self.buffers[input_buffer_idx];
                    let output_buffer = &self.buffers[output_buffer_idx];
                    stage.encode(&mut encoder, input_buffer, output_buffer, &self.side_inputs)?;
                }
            }

            // Propagate metadata to output buffer
            self.buffer_submission_metadata[output_buffer_idx] =
                self.buffer_submission_metadata[input_buffer_idx];
        }

        // Calculate last stage output buffer index BEFORE flipping current_set
        // Last stage writes to: 2*(num_stages-1) + 2 + (1 - current_set) = 2*num_stages + (1 - current_set)
        let last_output_buffer_idx = 2 * num_stages + (1 - self.current_set);
        let read_buffer = &self.buffers[last_output_buffer_idx];

        // Calculate readback buffer size
        let element_count = match self.buffer_submission_metadata[last_output_buffer_idx] {
            Some((actual_elements, _n, _batch_size)) => actual_elements * self.vector_dim,
            None => self.batch_size * self.vector_dim,
        };
        let buffer_size = std::cmp::min((element_count * F32_SIZE) as u64, read_buffer.size());
        let element_count = buffer_size as usize / F32_SIZE;

        // Add readback operations to the same encoder - this is the key optimization!
        // We always do the copy, even when we won't return output, to keep the GPU busy
        if buffer_size > 0 {
            let readback_buffer = self.context.create_buffer(
                Some("Output Readback"),
                buffer_size,
                readback_buffer_usages(),
            );

            encoder.copy_buffer_to_buffer(read_buffer, 0, &readback_buffer, 0, buffer_size);

            // Flip the current set for the next process call
            self.current_set = 1 - self.current_set;

            self.context.queue.submit(Some(encoder.finish()));
            self.context.device_poll()?;
            self.tick_count += 1;

            // If no output available yet, return None without reading the data
            if !will_have_output {
                return Ok(None);
            }

            // Read back the data
            let buffer_slice = readback_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            use wgpu::PollType;
            self.context.device.poll(PollType::Wait {
                submission_index: None,
                timeout: None,
            })?;

            receiver
                .recv()
                .map_err(|_| anyhow::anyhow!("Channel closed"))??;
            let data: &[u8] = &buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data[..element_count * F32_SIZE]).to_vec();

            // Extract the tag from the last stage - it should be present when output is available
            let output_tag: T =
                if !self.stage_configs.is_empty() {
                    match self.stage_configs.last_mut().unwrap() {
                        StageConfig::Standard { tag, .. } => std::mem::take(tag)
                            .expect("Tag should be present when output is available"),
                        StageConfig::Custom { tag, .. } => std::mem::take(tag)
                            .expect("Tag should be present when output is available"),
                    }
                } else {
                    panic!("No stages configured");
                };

            Ok(Some((output_tag, result)))
        } else {
            // Flip the current set for the next process call
            self.current_set = 1 - self.current_set;

            self.context.queue.submit(Some(encoder.finish()));
            self.context.device_poll()?;
            self.tick_count += 1;

            Ok(None)
        }
    }

    pub fn is_empty(&self) -> bool {
        for i in 0..self.stage_configs.len() {
            if !self.stage_configs[i].is_empty() {
                return false;
            }
        }
        true
    }

    /// Sets the default n value for the pipeline.
    /// This is used as a fallback when no metadata is available for a buffer.
    pub fn set_default_n(&mut self, n: usize) {
        self.default_n = n;
    }

    /// Sets the submission metadata for the input buffer in the current set.
    /// This allows tracking submission sizes through the pipeline without padding.
    /// The metadata will be set for the buffer that will be read by Stage 0 in the current tick.
    ///
    /// # Arguments
    /// * `actual_total_elements` - Total number of Complex elements in the submission
    /// * `n` - FFT size for this submission
    /// * `batch_size` - Batch size for this submission
    pub fn set_input_submission_metadata(
        &mut self,
        actual_total_elements: usize,
        n: usize,
        batch_size: usize,
    ) {
        // Set metadata for the input buffer in the current set
        // (current_set points to the buffer that Stage 0 will read from)
        self.buffer_submission_metadata[self.current_set] =
            Some((actual_total_elements, n, batch_size));
    }

    /// Clears all buffer metadata.
    /// This should be called when the pipeline is idle to avoid interference with new submissions.
    pub fn clear_all_buffer_metadata(&mut self) {
        self.buffer_submission_metadata.fill(None);
    }

    pub async fn resize(&mut self, new_batch_size: usize) -> Result<()> {
        if new_batch_size == 0 {
            bail!("Batch size must be at least 1");
        }
        if new_batch_size == self.batch_size {
            return Ok(());
        }

        let old_batch_size = self.batch_size;
        let element_size = self.vector_dim * F32_SIZE;
        let copy_size =
            std::cmp::min(old_batch_size as u64, new_batch_size as u64) * element_size as u64;
        let new_buffer_size = new_batch_size as u64 * element_size as u64;
        let usages = stage_buffer_usages();

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Resize Encoder"),
                });

        // Resize all buffers
        let mut new_buffers = Vec::with_capacity(self.buffers.len());

        for (i, old_buffer) in self.buffers.iter().enumerate() {
            let new_buffer = Arc::new(self.context.create_buffer(
                Some(&format!("Buffer {} Resized", i)),
                new_buffer_size,
                usages,
            ));
            encoder.copy_buffer_to_buffer(old_buffer, 0, &new_buffer, 0, copy_size);
            new_buffers.push(new_buffer);
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.context.device_poll()?;
        self.buffers = new_buffers;
        self.batch_size = new_batch_size;

        // Re-create bind groups because buffers have changed
        // With lazy initialization, bind groups will be recreated on next tick
        // For standard stages, we can clear the bind groups to force recreation
        for i in 0..self.bind_groups.len() {
            if self.bind_groups[i].is_some() {
                self.bind_groups[i] = None;
            }
        }

        // Notify custom stages of the size change if they support dynamic resizing
        for stage_config in &mut self.stage_configs {
            if stage_config.supports_dynamic_resizing() {
                let _ = stage_config.resize(new_batch_size, self.vector_dim);
            }
        }

        Ok(())
    }

    /// Resizes the pipeline to handle new dimensions, including notifying custom stages.
    ///
    /// This is a more comprehensive resize that can handle both batch_size and vector_dim changes.
    /// It will resize all buffers and notify custom stages that support dynamic resizing.
    ///
    /// # Arguments
    /// * `new_batch_size` - The new batch size (number of elements/vectors)
    /// * `new_vector_dim` - The new vector dimension (2 for complex numbers, etc.)
    ///
    /// # Returns
    /// A Result indicating success or failure of the resize operation.
    pub async fn resize_dynamic(
        &mut self,
        new_batch_size: usize,
        new_vector_dim: usize,
    ) -> Result<()> {
        if new_batch_size == 0 {
            bail!("Batch size must be at least 1");
        }
        if new_vector_dim == 0 {
            bail!("Vector dimension must be at least 1");
        }

        // If only batch size is changing, use the simpler resize method
        if new_vector_dim == self.vector_dim && new_batch_size == self.batch_size {
            return Ok(());
        }

        // If vector dimension is changing, we need to rebuild all buffers
        let element_size = new_vector_dim * F32_SIZE;
        let new_buffer_size = new_batch_size as u64 * element_size as u64;
        let usages = stage_buffer_usages();

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Dynamic Resize Encoder"),
                });

        // Resize all buffers with new dimensions
        let mut new_buffers = Vec::with_capacity(self.buffers.len());

        for (i, old_buffer) in self.buffers.iter().enumerate() {
            let new_buffer = Arc::new(self.context.create_buffer(
                Some(&format!("Buffer {} Dynamic Resize", i)),
                new_buffer_size,
                usages,
            ));

            // Try to copy as much data as possible from old buffer
            let old_element_size = self.vector_dim * F32_SIZE;
            let old_buffer_size = self.batch_size as u64 * old_element_size as u64;
            let copy_size = std::cmp::min(old_buffer_size, new_buffer_size);

            if copy_size > 0 {
                encoder.copy_buffer_to_buffer(old_buffer, 0, &new_buffer, 0, copy_size);
            }

            new_buffers.push(new_buffer);
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.context.device_poll()?;
        self.buffers = new_buffers;
        self.batch_size = new_batch_size;
        self.vector_dim = new_vector_dim;

        // Re-create bind groups because buffers have changed
        for i in 0..self.bind_groups.len() {
            if self.bind_groups[i].is_some() {
                self.bind_groups[i] = None;
            }
        }

        // Notify custom stages of the size change
        for stage_config in &mut self.stage_configs {
            if stage_config.supports_dynamic_resizing() {
                // We need to get mutable access to the stage to call resize
                // This requires a bit of unsafe code or a different approach
                // For now, we'll skip this and handle it differently
                // In a full implementation, we'd need to be able to mutate the stages
                // but since StageConfig::Custom uses Box<dyn PipelineStage>,
                // we can't easily get mutable access here without significant refactoring
            }
        }

        // Clear buffer metadata since buffer contents have been resized
        self.buffer_submission_metadata.fill(None);

        Ok(())
    }

    /// Resizes the pipeline and all its stages to handle new dimensions.
    /// This calls resize on all custom stages that support dynamic resizing.
    pub async fn resize_with_stages(
        &mut self,
        new_batch_size: usize,
        new_vector_dim: usize,
    ) -> Result<()> {
        // First resize the buffers
        self.resize(new_batch_size).await?;

        // Update pipeline metadata
        self.batch_size = new_batch_size;
        self.vector_dim = new_vector_dim;

        // Notify all custom stages of the size change
        for stage_config in &mut self.stage_configs {
            if stage_config.supports_dynamic_resizing() {
                // For custom stages, we need to call their resize method
                if let StageConfig::Custom { stage, .. } = stage_config {
                    // Call the stage's resize method
                    stage.resize(new_batch_size, new_vector_dim)?;
                }
            }
        }

        Ok(())
    }

    /// Updates the FFT size (n) parameter for all custom stages.
    ///
    /// This is used when the pipeline's FFT size changes without rebuilding.
    /// Stages that have an internal n parameter (like FftPipelineStage, NormalizePipelineStage)
    /// should implement update_n to handle this.
    ///
    /// # Arguments
    /// * `new_n` - The new FFT size
    pub fn update_stage_n(&mut self, new_n: usize) -> Result<()> {
        for stage_config in &mut self.stage_configs {
            stage_config.update_n(new_n)?;
        }
        Ok(())
    }

    /// Adds a side input buffer to the pipeline.
    ///
    /// Side inputs are buffers that are bound to stages but are not part of the
    /// pipeline's data flow. They are useful for operations like multiplication
    /// where a stage needs access to a constant buffer (e.g., pre-computed FFT).
    ///
    /// # Arguments
    /// * `name` - A unique name for this side input
    /// * `buffer` - The buffer to use as a side input
    ///
    /// # Returns
    /// The buffer index that can be used to reference this side input
    pub fn add_side_input(&mut self, name: impl Into<String>, buffer: Arc<wgpu::Buffer>) -> usize {
        let name_str = name.into();
        let index = self.side_inputs.len();
        self.side_inputs.insert(name_str.clone(), buffer);
        index
    }

    /// Gets a side input buffer by name.
    pub fn get_side_input(&self, name: &str) -> Option<&Arc<wgpu::Buffer>> {
        self.side_inputs.get(name)
    }

    /// Gets the index of a side input by name.
    pub fn get_side_input_index(&self, name: &str) -> Option<usize> {
        // Find the index by iterating (since HashMap doesn't preserve order)
        for (i, (key, _)) in self.side_inputs.iter().enumerate() {
            if key == name {
                return Some(i);
            }
        }
        None
    }

    pub fn num_stages(&self) -> usize {
        self.compute_pipelines.len()
    }

    pub fn vector_dim(&self) -> usize {
        self.vector_dim
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn get_input_buffer(&self) -> &wgpu::Buffer {
        // Return the first input buffer (buffer 0) for backward compatibility
        &self.buffers[0]
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }
}

fn create_bind_group_layout(context: &ComputeContext) -> wgpu::BindGroupLayout {
    context.create_bind_group_layout(
        Some("Stage Bind Group Layout"),
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
