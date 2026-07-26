//! WGPU utilities and context management for the pipeline.

use anyhow::{Context, Result};
use wgpu::{Adapter, Device, Instance, Queue};

/// WGPU compute context that manages the device, queue, and adapter.
///
/// This provides a centralized way to create and manage WGPU resources
/// for compute-only workloads (no graphics/surface).
#[derive(Debug)]
pub struct ComputeContext {
    /// The WGPU instance
    pub instance: Instance,
    /// The selected adapter
    pub adapter: Adapter,
    /// The logical device
    pub device: Device,
    /// The command queue
    pub queue: Queue,
}

impl ComputeContext {
    /// Creates a new compute context with the best available adapter.
    ///
    /// # Arguments
    ///
    /// * `power_preference` - Preferred power profile (LowPower vs HighPerformance)
    ///
    /// # Returns
    ///
    /// A new `ComputeContext` with device, queue, and adapter.
    pub async fn new(power_preference: wgpu::PowerPreference) -> Result<Self> {
        // Create WGPU instance
        let instance = wgpu::Instance::default();

        // Request adapter with compute capability
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("No suitable GPU adapter found")?;

        // Verify compute features
        let adapter_info = adapter.get_info();
        log::info!(
            "Selected adapter: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        // Request device with compute features
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Ping-Pong Pipeline Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .context("Failed to create WGPU device")?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Creates a new compute context with high performance preference.
    pub async fn new_high_performance() -> Result<Self> {
        Self::new(wgpu::PowerPreference::HighPerformance).await
    }

    /// Creates a new compute context with low power preference.
    pub async fn new_low_power() -> Result<Self> {
        Self::new(wgpu::PowerPreference::LowPower).await
    }

    /// Creates a new buffer with the specified usage and size.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional label for debugging
    /// * `size` - Size of the buffer in bytes
    /// * `usage` - Buffer usage flags
    ///
    /// # Returns
    ///
    /// A new WGPU buffer.
    pub fn create_buffer(
        &self,
        label: Option<&str>,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Creates a new buffer initialized with data.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional label for debugging
    /// * `data` - Data to initialize the buffer with
    /// * `usage` - Buffer usage flags
    ///
    /// # Returns
    ///
    /// A new WGPU buffer with the data.
    pub fn create_buffer_with_data<T: bytemuck::Pod>(
        &self,
        label: Option<&str>,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let buffer = self.create_buffer(label, std::mem::size_of_val(data) as u64, usage);
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    }

    /// Creates a compute pipeline from WGSL source code.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional label for debugging
    /// * `wgsl_source` - The WGSL shader source code
    /// * `bind_group_layouts` - Bind group layouts used by the shader
    ///
    /// # Returns
    ///
    /// A new compute pipeline.
    pub fn create_compute_pipeline(
        &self,
        label: Option<&str>,
        wgsl_source: &str,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> Result<wgpu::ComputePipeline> {
        let shader_label = label.map(|l| format!("{l} Shader"));
        let shader_label_ref = shader_label.as_deref();
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: shader_label_ref,
                source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
            });

        let pipeline_layout_label = label.map(|l| format!("{l} Pipeline Layout"));
        let pipeline_layout_label_ref = pipeline_layout_label.as_deref();
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: pipeline_layout_label_ref,
                bind_group_layouts: &bind_group_layouts
                    .iter()
                    .map(|&lg| Some(lg))
                    .collect::<Vec<_>>(),
                immediate_size: 0,
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label,
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Ok(compute_pipeline)
    }

    /// Creates a bind group layout for compute shaders.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional label for debugging
    /// * `entries` - Bind group layout entries
    ///
    /// # Returns
    ///
    /// A new bind group layout.
    pub fn create_bind_group_layout(
        &self,
        label: Option<&str>,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> wgpu::BindGroupLayout {
        self.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label, entries })
    }

    /// Creates a bind group.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional label for debugging
    /// * `layout` - The bind group layout
    /// * `entries` - Bind group entries (resources to bind)
    ///
    /// # Returns
    ///
    /// A new bind group.
    pub fn create_bind_group(
        &self,
        label: Option<&str>,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry],
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries,
        })
    }

    /// Waits for the GPU to complete all submitted work.
    ///
    /// This is useful for synchronization points like resizing.
    pub fn device_poll(&self) -> Result<()> {
        use wgpu::PollType;
        // Wait indefinitely for the most recent submission
        self.device.poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        Ok(())
    }
}

/// Default buffer usages for stage buffers in the pipeline.
///
/// Stage buffers need to be:
/// - Storage buffers for compute shader access
/// - Copy source for reading data back
/// - Copy destination for writing data in
/// - Uniform buffer usage is NOT included (use storage instead)
pub fn stage_buffer_usages() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}

/// Buffer usages for staging buffers (CPU to GPU uploads).
pub fn staging_buffer_usages() -> wgpu::BufferUsages {
    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE
}

/// Buffer usages for readback buffers (GPU to CPU downloads).
pub fn readback_buffer_usages() -> wgpu::BufferUsages {
    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ
}
