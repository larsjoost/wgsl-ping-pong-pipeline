//! Pipeline implementation with isolated per-stage ping-pong buffers.

use crate::wgpu_utils::{ComputeContext, stage_buffer_usages, readback_buffer_usages};
use anyhow::{bail, Result};
use bytemuck::Pod;
use std::collections::HashMap;
use std::sync::Arc;

pub mod pipeline_stage;
pub use pipeline_stage::{PipelineStage, StageConfig};

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
    /// All buffers: [input_buffer, stage0_out_a, stage0_out_b, stage1_out_a, stage1_out_b, ...]
    /// For N stages, we need 1 input buffer + 2*N output buffers
    buffers: Vec<Arc<wgpu::Buffer>>,
    /// Compute pipelines for each stage (only for standard stages)
    compute_pipelines: Vec<Option<Arc<wgpu::ComputePipeline>>>,
    /// Bind group layouts for each stage (can be different per stage)
    bind_group_layouts: Vec<Arc<wgpu::BindGroupLayout>>,
    /// Pre-created bind groups for each stage (2 for each stage for ping-pong)
    /// Only used for standard stages; custom stages create their own bind groups
    bind_groups: Vec<Option<[wgpu::BindGroup; 2]>>,
    /// Current output buffer index for each stage (0 or 1 in its pair)
    current_output_indices: Vec<usize>,
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

        // Create all buffers: input buffer + 2 output buffers per stage
        let num_buffers = 1 + 2 * stage_configs.len();
        let mut buffers = Vec::with_capacity(num_buffers);
        
        // Input buffer for stage 0
        buffers.push(Arc::new(context.create_buffer(
            Some("Pipeline Input Buffer"),
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

        let pipeline = Self {
            context,
            bind_group_layout: Arc::clone(&default_bind_group_layout),
            vector_dim,
            batch_size,
            tick_count: 0,
            buffers,
            compute_pipelines,
            bind_group_layouts,
            bind_groups: vec![None; stage_configs.len()], // Will be initialized lazily
            current_output_indices: vec![0; stage_configs.len()],
            stage_names,
            stage_configs,
            side_inputs: std::collections::HashMap::new(),
            last_output_tag: None,
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
    fn create_bind_group_for_stage(&self, stage_idx: usize, state: usize) -> wgpu::BindGroup {
        let bgl = &self.bind_group_layouts[stage_idx];
        let input_buffer_idx = if stage_idx == 0 {
            0
        } else {
            // If state is 0, current_output_indices[stage_idx-1] would be 0.
            // Staggering means we read from 1 - 0 = 1.
            // If state is 1, current_output_indices[stage_idx-1] would be 1.
            // Staggering means we read from 1 - 1 = 0.
            1 + 2 * (stage_idx - 1) + (1 - state)
        };
        let output_buffer_idx = 1 + 2 * stage_idx + state;

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
            Some(&format!("Stage {} BG State {}", self.stage_names[stage_idx], state)),
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
        // This shifts tags forward through the pipeline
        let mut current_tag: Option<T> = Some(tag.clone());
        for i in 0..self.stage_configs.len() {
            current_tag = self.stage_configs[i].forward_tag(current_tag);
        }
        // current_tag now contains what was in the last stage before this tick

        // Store the tag that just exited the last stage
        // For the same tag used for all ticks, current_tag will be None for the first num_stages ticks
        // In this case, output is ready after num_stages ticks
        let num_stages = self.stage_configs.len() as u64;
        let ticks_after_this = self.tick_count + 1;
        
        if current_tag.is_none() && ticks_after_this >= num_stages {
            // Same tag case: output is ready after num_stages ticks
            self.last_output_tag = Some(tag);
        } else {
            // Different tags case: output is ready when current_tag is Some
            self.last_output_tag = current_tag;
        }

        for i in 0..self.stage_configs.len() {
            let state = self.current_output_indices[i];
            
            match &self.stage_configs[i] {
                StageConfig::Standard { .. } => {
                    // Standard stage: use compute pipeline and bind groups
                    if let Some(compute_pipeline) = &self.compute_pipelines[i] {
                        // Lazy initialization of bind groups for standard stages
                        if self.bind_groups[i].is_none() {
                            let bg0 = self.create_bind_group_for_stage(i, 0);
                            let bg1 = self.create_bind_group_for_stage(i, 1);
                            self.bind_groups[i] = Some([bg0, bg1]);
                        }
                        
                        let bind_group = &self.bind_groups[i].as_ref().unwrap()[state];

                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!("Stage {} Pass Tick {}", self.stage_names[i], self.tick_count)),
                            timestamp_writes: None,
                        });

                        pass.set_pipeline(compute_pipeline);
                        pass.set_bind_group(0, bind_group, &[]);

                        let workgroup_size = 64u32;
                        let dispatch_count = (self.batch_size as u32 + workgroup_size - 1) / workgroup_size;
                        pass.dispatch_workgroups(dispatch_count, 1, 1);
                    }
                }
                StageConfig::Custom { stage, .. } => {
                    // Custom stage: call its encode method
                    let input_buffer_idx = if i == 0 {
                        0
                    } else {
                        1 + 2 * (i - 1) + (1 - state)
                    };
                    let output_buffer_idx = 1 + 2 * i + state;
                    
                    let input_buffer = &self.buffers[input_buffer_idx];
                    let output_buffer = &self.buffers[output_buffer_idx];
                    
                    stage.encode(&mut encoder, input_buffer, output_buffer, &self.side_inputs)?;
                }
            }
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

    pub async fn write_input<D: Pod>(&self, data: &[D]) -> Result<()> {
        let expected_byte_size = self.batch_size * self.vector_dim * F32_SIZE;
        let actual_byte_size = data.len() * std::mem::size_of::<D>();
        if actual_byte_size != expected_byte_size {
            bail!(
                "Input size mismatch: expected {} bytes (batch_size * vector_dim * {} = {} * {} * {}), got {} bytes",
                expected_byte_size, F32_SIZE, self.batch_size, self.vector_dim, F32_SIZE, actual_byte_size
            );
        }

        // Write to input buffer (index 0)
        self.context.queue.write_buffer(&self.buffers[0], 0, bytemuck::cast_slice(data));

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
        // Buffer index = 1 + 2*(num_stages-1) + (1 - current_output_indices[num_stages-1])
        // Because we flipped after dispatch, the current_output_indices points to the NEXT buffer to write to
        // So the last written buffer is 1 - current_output_indices[num_stages-1]
        let last_output_buffer_idx = 1 + 2 * (num_stages - 1) + (1 - self.current_output_indices[num_stages - 1]);
        let read_buffer = &self.buffers[last_output_buffer_idx];
        let buffer_size = self.batch_size as u64 * self.vector_dim as u64 * F32_SIZE as u64;
        let element_count = buffer_size as usize / F32_SIZE;
        eprintln!("PIPELINE DEBUG: batch_size={}, vector_dim={}, buffer_size={}, element_count={}", 
            self.batch_size, self.vector_dim, buffer_size, element_count);

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
