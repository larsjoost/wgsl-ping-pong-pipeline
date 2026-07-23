//! Pipeline implementation with isolated per-stage ping-pong buffers.

use crate::wgpu_utils::{ComputeContext, stage_buffer_usages, readback_buffer_usages};
use anyhow::{bail, Result};
use bytemuck::Pod;
use std::collections::HashMap;
use std::sync::Arc;

pub mod pipeline_stage;
pub mod variable_size;
pub use pipeline_stage::{PipelineStage, StageConfig};
pub use variable_size::{VariableSizePipeline, VariableSizePipelineBuilder, StageSizeConfig};

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
    pub fn new(name: impl Into<String>, wgsl: impl Into<String>, vector_dim: usize, batch_size: usize) -> Self {
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
    format!(
        r#"
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let idx = id.x;
    if (idx >= arrayLength(&input)) {{ return; }}
    output[idx] = input[idx];
}}
"#
    )
}

/// Pipeline builder.
#[derive(Debug, Default)]
pub struct PipelineBuilder<T> {
    stages: Vec<StageConfig<T>>,
    context: Option<Arc<ComputeContext>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Clone> PipelineBuilder<T> {
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

    pub fn identity(mut self, name: impl Into<String>, vector_dim: usize, batch_size: usize) -> Self {
        self.stages.push(StageConfig::Standard { stage: Stage::identity(name, vector_dim, batch_size), tag: None });
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
    /// All buffers: [input_buffer_a, input_buffer_b, stage0_out_a, stage0_out_b, stage1_out_a, stage1_out_b, ...]
    /// For N stages, we need 2 input buffers + 2*N output buffers
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
    /// Only used for standard stages; custom stages create their own bind groups
    bind_groups: Vec<Option<[[wgpu::BindGroup; 2]; 2]>>,
    /// Current output buffer index for each stage (0 or 1 in its pair)
    current_output_indices: Vec<usize>,
    /// Current input buffer index to write to (0 or 1) for ping-pong input
    current_input_write_index: usize,
    /// Stage names
    stage_names: Vec<String>,
    /// Stage configurations (standard or custom)
    stage_configs: Vec<StageConfig<T>>,
    /// Side input buffers managed by the pipeline
    /// Maps from name to buffer
    side_inputs: HashMap<String, Arc<wgpu::Buffer>>,
    /// The tag from the last tick() call's input
    /// When this is None, it means no tick() has been called yet
    last_output_tag: Option<T>,
    /// Per-buffer metadata: for each buffer, store submission metadata
    /// This allows tracking submission sizes through the pipeline without padding
    /// Index corresponds to buffer index in the buffers vec
    /// Stores (actual_total_elements, n, batch_size) for the submission in that buffer
    buffer_submission_metadata: Vec<Option<(usize, usize, usize)>>,
    /// Default n value to use when no metadata is available for a buffer
    /// This is set when the pipeline is created or resized
    default_n: usize,
}

impl<T: Clone> Pipeline<T> {
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
                        stage.name(), stage.vector_dim(), vector_dim
                    );
                }
                if stage.batch_size() != batch_size {
                    bail!(
                        "All stages must have the same batch size. Stage '{}' has {}, expected {}",
                        stage.name(), stage.batch_size(), batch_size
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
        for (_i, stage_config) in stage_configs.iter().enumerate() {
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
        let mut compute_pipelines: Vec<Option<Arc<wgpu::ComputePipeline>>> = Vec::with_capacity(stage_configs.len());
        let mut bind_group_layouts = Vec::with_capacity(stage_configs.len());
        let mut stage_names = Vec::with_capacity(stage_configs.len());
        let mut bind_groups: Vec<Option<[wgpu::BindGroup; 2]>> = Vec::with_capacity(stage_configs.len());
        
        // Initialize custom stages in place (we take ownership of stage_configs)
        for stage_config in &mut stage_configs {
            if let StageConfig::Custom { stage, .. } = stage_config {
                if stage.requires_initialization() {
                    stage.initialize(&context)?;
                }
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
            current_output_indices: vec![0; stage_configs.len()],
            current_input_write_index: 0,
            stage_names,
            stage_configs,
            side_inputs: std::collections::HashMap::new(),
            last_output_tag: None,
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
    fn create_bind_group_for_stage(&self, stage_idx: usize, input_buffer_variant: usize, output_state: usize) -> wgpu::BindGroup {
        let bgl = &self.bind_group_layouts[stage_idx];
        let input_buffer_idx = if stage_idx == 0 {
            // Stage 0: input_buffer_variant selects between the 2 input buffers
            input_buffer_variant
        } else {
            // For stage i > 0: input comes from previous stage's output
            // input_buffer_variant is used to select between the 2 output buffers of the previous stage
            2 + 2 * (stage_idx - 1) + input_buffer_variant
        };
        let output_buffer_idx = 2 + 2 * stage_idx + output_state;

        // Check how many bindings this stage's layout has
        // If it has 3 bindings (like multiply stage), we need to provide 3 entries
        // If it has 2 bindings (default), we provide 2 entries
        // Check the number of bindings by examining the entries
        // We need to access the BindGroupLayout's entries, but it's not directly accessible
        // For now, we'll use a heuristic: check if this stage has a custom layout
        let stage_has_custom_layout = Arc::as_ptr(&self.bind_group_layouts[stage_idx]) != Arc::as_ptr(&self.bind_group_layout);
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
            Some(&format!("Stage {} BG Input {} Output {}", self.stage_names[stage_idx], input_buffer_variant, output_state)),
            bgl,
            &entries,
        )
    }

    pub async fn tick(&mut self, tag: T) -> Result<()> {
        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Tick {} Encoder", self.tick_count)),
            },
        );

        // Process tag flow through pipeline
        // Insert tag into first stage, get its previous tag
        // Then forward that tag to next stage, and so on
        // In immediate mode, all stages process in a single tick, so output is ready immediately
        let mut current_tag: Option<T> = Some(tag.clone());
        for i in 0..self.stage_configs.len() {
            current_tag = self.stage_configs[i].forward_tag(current_tag);
        }
        
        // In immediate mode, output is ready after each tick
        self.last_output_tag = Some(tag);

        for i in 0..self.stage_configs.len() {
            let state = self.current_output_indices[i];
            
            // Calculate input buffer index for this stage
            // In immediate mode, all stages process in a single tick in sequence
            // Stage 0 reads from the input buffer, each subsequent stage reads from the previous stage's output buffer
            let input_buffer_idx = if i == 0 {
                // Stage 0 reads from the input buffer that has data
                1 - self.current_input_write_index
            } else {
                // In immediate mode: read from the buffer that the previous stage wrote to in the CURRENT tick
                // The previous stage (i-1) writes to buffer: 2 + 2*(i-1) + current_output_indices[i-1]
                // So we read from that same buffer
                2 + 2 * (i - 1) + self.current_output_indices[i - 1]
            };
            
            let output_buffer_idx = 2 + 2 * i + state;
            
            // Get the submission metadata from the input buffer and propagate to stage
            // If no metadata is available, use default values
            let metadata = self.buffer_submission_metadata[input_buffer_idx];
            let (actual_elements, n) = match metadata {
                Some((actual_elements, n, _batch_size)) => (actual_elements, n),
                None => {
                    // No metadata for this buffer, use default values
                    // This happens when the pipeline hasn't been filled yet or data hasn't reached this stage
                    // Use the pipeline's default values
                    (self.batch_size, self.default_n)
                }
            };
            
            // Update the stage with the actual elements and n from its input submission
            self.stage_configs[i].update_actual_total_elements(actual_elements)?;
            self.stage_configs[i].update_n(n)?;
            
            match &self.stage_configs[i] {
                StageConfig::Standard { .. } => {
                    // Standard stage: use compute pipeline and bind groups
                    if let Some(compute_pipeline) = &self.compute_pipelines[i] {
                        // Lazy initialization of bind groups for standard stages
                        // We need 4 bind groups: 2 input variants × 2 output states
                        if self.bind_groups[i].is_none() {
                            let bgs = [[self.create_bind_group_for_stage(i, 0, 0), self.create_bind_group_for_stage(i, 0, 1)],
                                       [self.create_bind_group_for_stage(i, 1, 0), self.create_bind_group_for_stage(i, 1, 1)]];
                            self.bind_groups[i] = Some(bgs);
                        }
                        
                        // Select the appropriate bind group based on input buffer variant and output state
                        let (input_buffer_variant, output_state) = if i == 0 {
                            // For stage 0, input_buffer_variant is determined by current_input_write_index
                            // The buffer with data is 1 - current_input_write_index
                            (1 - self.current_input_write_index, state)
                        } else {
                            // For other stages, input_buffer_variant is determined by staggering
                            // At the start of tick, state = the buffer index this stage will write to
                            // In the previous tick, the previous stage had state = 1 - state
                            // So the previous stage wrote to buffer (1 - state)
                            (1 - state, state)
                        };
                        let bind_group = &self.bind_groups[i].as_ref().unwrap()[input_buffer_variant][output_state];

                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!("Stage {} Pass Tick {}", self.stage_names[i], self.tick_count)),
                            timestamp_writes: None,
                        });

                        pass.set_pipeline(compute_pipeline);
                        pass.set_bind_group(0, bind_group, &[]);

                        let workgroup_size = 64u32;
                        let dispatch_count = (actual_elements as u32 + workgroup_size - 1) / workgroup_size;
                        pass.dispatch_workgroups(dispatch_count, 1, 1);
                    }
                }
                StageConfig::Custom { stage, .. } => {
                    // Custom stage: call its encode method
                    // Reuse input_buffer_idx calculated above
                    let input_buffer = &self.buffers[input_buffer_idx];
                    let output_buffer = &self.buffers[output_buffer_idx];
                    
                    stage.encode(&mut encoder, input_buffer, output_buffer, &self.side_inputs)?
                }
            }
            
            // Propagate submission metadata to output buffer
            // The output buffer is being written with data from the input buffer in this tick,
            // so it should have the same metadata as the input buffer
            self.buffer_submission_metadata[output_buffer_idx] = self.buffer_submission_metadata[input_buffer_idx];
        }

        // Flip all output indices
        for i in 0..self.current_output_indices.len() {
            self.current_output_indices[i] = 1 - self.current_output_indices[i];
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.context.device_poll()?;
        self.tick_count += 1;

        Ok(())
    }

    /// Sets the default n value for the pipeline.
    /// This is used as a fallback when no metadata is available for a buffer.
    pub fn set_default_n(&mut self, n: usize) {
        self.default_n = n;
    }
    
    /// Sets the submission metadata for the next input buffer to be written.
    /// This allows tracking submission sizes through the pipeline without padding.
    /// The metadata will be set for the buffer that will be written to by the next write_input() call.
    /// 
    /// # Arguments
    /// * `actual_total_elements` - Total number of Complex elements in the submission
    /// * `n` - FFT size for this submission
    /// * `batch_size` - Batch size for this submission
    pub fn set_input_submission_metadata(&mut self, actual_total_elements: usize, n: usize, batch_size: usize) {
        // Set metadata for the buffer that will be written to next
        // (current_input_write_index points to the next buffer to write to)
        self.buffer_submission_metadata[self.current_input_write_index] = Some((actual_total_elements, n, batch_size));
    }
    
    /// Clears all buffer metadata.
    /// This should be called when the pipeline is idle to avoid interference with new submissions.
    pub fn clear_all_buffer_metadata(&mut self) {
        for metadata in &mut self.buffer_submission_metadata {
            *metadata = None;
        }
    }

    pub async fn write_input<D: Pod>(&mut self, data: &[D]) -> Result<()> {
        let actual_byte_size = data.len() * std::mem::size_of::<D>();
        if actual_byte_size == 0 {
            bail!("Input data cannot be empty");
        }

        let element_size = self.vector_dim * F32_SIZE;
        if actual_byte_size % element_size != 0 {
            bail!(
                "Input byte size {} is not a multiple of element size {} (vector_dim {} * F32_SIZE {})",
                actual_byte_size, element_size, self.vector_dim, F32_SIZE
            );
        }

        let actual_batch_size = actual_byte_size / element_size;
        let write_idx = self.current_input_write_index;

        if actual_byte_size as u64 > self.buffers[write_idx].size() {
            let new_batch_size = std::cmp::max(actual_batch_size, self.batch_size);
            self.resize(new_batch_size).await?;
        }

        // Write to the current input buffer
        self.context.queue.write_buffer(&self.buffers[write_idx], 0, bytemuck::cast_slice(data));
        
        let custom_n = match self.buffer_submission_metadata[write_idx] {
            Some((_, custom_n, _)) => custom_n,
            None => self.default_n,
        };
        self.buffer_submission_metadata[write_idx] = Some((actual_batch_size, custom_n, actual_batch_size));

        // Toggle write index for next submission
        self.current_input_write_index = 1 - write_idx;

        Ok(())
    }

    pub async fn read_output(&self) -> Result<Option<(Option<T>, Vec<f32>)>> {
        // Return None if no tick has been called yet (tag is None)
        if self.last_output_tag.is_none() {
            return Ok(None);
        }

        let num_stages = self.compute_pipelines.len();
        if num_stages == 0 {
            bail!("Pipeline has no stages");
        }

        // Output is in the last stage's current output buffer
        // Buffer index = 2 + 2*(num_stages-1) + (1 - current_output_indices[num_stages-1])
        // Because we flipped after dispatch, the current_output_indices points to the NEXT buffer to write to
        // So the last written buffer is 1 - current_output_indices[num_stages-1]
        // Note: We now have 2 input buffers, so stage outputs start at index 2
        let last_output_buffer_idx = 2 + 2 * (num_stages - 1) + (1 - self.current_output_indices[num_stages - 1]);
        let read_buffer = &self.buffers[last_output_buffer_idx];

        let element_count = match self.buffer_submission_metadata[last_output_buffer_idx] {
            Some((actual_elements, _n, _batch_size)) => actual_elements * self.vector_dim,
            None => self.batch_size * self.vector_dim,
        };
        let buffer_size = std::cmp::min(
            (element_count * F32_SIZE) as u64,
            read_buffer.size(),
        );
        let element_count = buffer_size as usize / F32_SIZE;

        let readback_buffer = self.context.create_buffer(
            Some("Output Readback"),
            buffer_size,
            readback_buffer_usages(),
        );

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });

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

        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Resize Encoder"),
            },
        );

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
    pub async fn resize_dynamic(&mut self, new_batch_size: usize, new_vector_dim: usize) -> Result<()> {
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
        
        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Dynamic Resize Encoder"),
            },
        );

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
        for metadata in &mut self.buffer_submission_metadata {
            *metadata = None;
        }

        Ok(())
    }
    
    /// Resizes the pipeline and all its stages to handle new dimensions.
    /// This calls resize on all custom stages that support dynamic resizing.
    pub async fn resize_with_stages(&mut self, new_batch_size: usize, new_vector_dim: usize) -> Result<()> {
        // First resize the buffers
        self.resize(new_batch_size).await?;
        
        // Update pipeline metadata
        self.batch_size = new_batch_size;
        self.vector_dim = new_vector_dim;
        
        // Notify all custom stages of the size change
        for stage_config in &mut self.stage_configs {
            if stage_config.supports_dynamic_resizing() {
                // For custom stages, we need to call their resize method
                match stage_config {
                    StageConfig::Custom { stage, .. } => {
                        // Call the stage's resize method
                        stage.resize(new_batch_size, new_vector_dim)?;
                    }
                    _ => {}
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
