//! Pipeline stage trait for custom stage implementations.
//!
//! This module provides the `PipelineStage` trait which allows external libraries
//! like `wgsl-fft` to provide custom encoding logic that can be integrated into
//! the ping-pong pipeline.

use anyhow::Result;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use wgpu::CommandEncoder;

use crate::wgpu_utils::ComputeContext;

/// A pipeline stage that can encode its own compute operations.
///
/// This trait allows stages to provide custom encoding logic, which is necessary
/// for operations like FFT that require multiple compute passes (bit-reversal
/// + butterfly stages) and cannot be expressed as a single WGSL shader.
///
/// # Example
///
/// ```ignore
/// use wgsl_ping_pong_pipeline::pipeline::PipelineStage;
///
/// struct MyFftStage {
///     n: usize,
///     batch_size: u32,
///     fft_pipelines: Arc<wgsl_fft::FftPipelines>,
/// }
///
/// impl PipelineStage for MyFftStage {
///     fn name(&self) -> &str { "my_fft" }
///     fn vector_dim(&self) -> usize { 2 }
///     fn batch_size(&self) -> usize { self.batch_size as usize }
///     
///     fn encode(&self, encoder: &mut CommandEncoder, input: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<()> {
///         self.fft_pipelines.encode_fft(
///             encoder,
///             self.n,
///             self.batch_size,
///             wgsl_fft::FftDirection::Forward,
///             input,
///             output,
///         );
///         Ok(())
///     }
///     
///     fn initialize(&mut self, _context: &ComputeContext) -> Result<()> {
///         // Already initialized in new()
///         Ok(())
///     }
///     
///     fn requires_initialization(&self) -> bool {
///         false
///     }
/// }
/// ```
pub trait PipelineStage: Debug + Send + Sync {
    /// Returns the name of this stage for debugging and identification.
    fn name(&self) -> &str;

    /// Returns the vector dimension (2 for complex numbers, etc.).
    fn vector_dim(&self) -> usize;

    /// Returns the batch size (number of elements/vectors).
    fn batch_size(&self) -> usize;

    /// Returns the names of any side input buffers this stage requires.
    ///
    /// Side inputs are buffers that are not part of the pipeline's data flow
    /// but are needed by this stage (e.g., a constant buffer for multiplication).
    /// The pipeline will provide these buffers when encoding.
    fn side_input_names(&self) -> Vec<&str> {
        Vec::new()
    }

    /// Encodes the compute operations for this stage into the given encoder.
    ///
    /// This method is called during the pipeline's tick() to encode the stage's
    /// compute operations. It receives the input and output buffers and should
    /// encode all necessary compute passes into the encoder.
    ///
    /// # Arguments
    /// * `encoder` - The command encoder to write compute operations to
    /// * `input` - The input buffer (binding 0 for most stages)
    /// * `output` - The output buffer (binding 1 for most stages)
    /// * `side_inputs` - Optional side input buffers indexed by name
    ///
    /// # Returns
    /// A Result indicating success or failure of the encoding operation.
    fn encode(
        &self,
        encoder: &mut CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        side_inputs: &HashMap<String, Arc<wgpu::Buffer>>,
    ) -> Result<()>;

    /// Initializes the stage with the pipeline's compute context.
    ///
    /// This is called when the pipeline is built, allowing the stage to
    /// access the device and queue. For stages that need to create GPU
    /// resources (like FftPipelines), this is where that initialization happens.
    ///
    /// # Arguments
    /// * `context` - The compute context containing device and queue
    ///
    /// # Returns
    /// A Result indicating success or failure of the initialization.
    fn initialize(&mut self, context: &ComputeContext) -> Result<()>;

    /// Returns true if this stage requires initialization.
    ///
    /// Stages that don't need any GPU context (e.g., simple shaders) can return false.
    fn requires_initialization(&self) -> bool {
        true
    }

    /// Resizes this stage to handle new dimensions.
    ///
    /// This method is called when the pipeline needs to handle data of a different size.
    /// Stages that support dynamic resizing should update their internal state
    /// (like batch_size, vector_dim) and recreate any GPU resources that depend on size.
    ///
    /// # Arguments
    /// * `new_batch_size` - The new batch size (number of elements/vectors)
    /// * `new_vector_dim` - The new vector dimension (2 for complex numbers, etc.)
    ///
    /// # Returns
    /// A Result indicating success or failure of the resize operation.
    /// Default implementation returns an error - stages that can resize should override this.
    fn resize(&mut self, _new_batch_size: usize, _new_vector_dim: usize) -> Result<()> {
        anyhow::bail!("Stage does not support dynamic resizing")
    }

    /// Returns true if this stage supports dynamic resizing.
    ///
    /// Stages that can handle size changes without full rebuild should return true.
    /// Default implementation returns false.
    fn supports_dynamic_resizing(&self) -> bool {
        false
    }

    /// Updates the FFT size parameter for stages that need it.
    ///
    /// This is called when the pipeline's FFT size (n) changes but the pipeline
    /// is not being rebuilt. Stages that use an internal n parameter for FFT
    /// operations should update it here.
    ///
    /// # Arguments
    /// * `new_n` - The new FFT size
    ///
    /// # Returns
    /// A Result indicating success or failure. Default implementation does nothing.
    fn update_n(&mut self, _new_n: usize) -> Result<()> {
        Ok(())
    }

    /// Updates the actual data size parameter for stages that need it.
    ///
    /// This is called to track the actual number of elements in a submission,
    /// which may be smaller than the pipeline's buffer size.
    ///
    /// # Arguments
    /// * `total_elements` - The actual number of elements (Complex values) in the submission
    ///
    /// # Returns
    /// A Result indicating success or failure. Default implementation does nothing.
    fn update_actual_total_elements(&mut self, _total_elements: usize) -> Result<()> {
        Ok(())
    }
}

/// Enum representing either a standard Stage or a custom PipelineStage.
#[derive(Debug)]
pub enum StageConfig<T> {
    /// A standard stage with a single WGSL shader and optional tag
    Standard {
        stage: crate::pipeline::Stage,
        tag: Option<T>,
    },
    /// A custom stage implementing PipelineStage trait with optional tag
    Custom {
        stage: Box<dyn PipelineStage>,
        tag: Option<T>,
    },
}

impl<T> StageConfig<T> {
    pub fn name(&self) -> &str {
        match self {
            StageConfig::Standard { stage, .. } => &stage.name,
            StageConfig::Custom { stage, .. } => stage.name(),
        }
    }

    pub fn forward_tag(&mut self, new_tag: Option<T>) -> Option<T> {
        match self {
            StageConfig::Standard { tag, .. } => std::mem::replace(tag, new_tag),
            StageConfig::Custom { tag, .. } => std::mem::replace(tag, new_tag),
        }
    }

    pub fn vector_dim(&self) -> usize {
        match self {
            StageConfig::Standard { stage, .. } => stage.vector_dim,
            StageConfig::Custom { stage, .. } => stage.vector_dim(),
        }
    }

    pub fn batch_size(&self) -> usize {
        match self {
            StageConfig::Standard { stage, .. } => stage.batch_size,
            StageConfig::Custom { stage, .. } => stage.batch_size(),
        }
    }

    pub fn element_size(&self) -> usize {
        self.vector_dim() * super::F32_SIZE
    }

    pub fn buffer_size(&self) -> u64 {
        self.batch_size() as u64 * self.element_size() as u64
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            StageConfig::Standard { stage, .. } => stage.validate(),
            StageConfig::Custom { stage, .. } => {
                if stage.vector_dim() == 0 {
                    anyhow::bail!("Vector dimension must be at least 1");
                }
                if stage.batch_size() == 0 {
                    anyhow::bail!("Batch size must be at least 1");
                }
                Ok(())
            }
        }
    }

    pub fn is_custom(&self) -> bool {
        matches!(self, StageConfig::Custom { .. })
    }

    pub fn as_custom(&self) -> Option<&dyn PipelineStage> {
        match self {
            StageConfig::Custom { stage, .. } => Some(stage.as_ref()),
            _ => None,
        }
    }

    pub fn as_custom_mut(&mut self) -> Option<&mut dyn PipelineStage> {
        match self {
            StageConfig::Custom { stage, .. } => Some(stage.as_mut()),
            _ => None,
        }
    }

    /// Returns a reference to the tag stored in this stage config
    pub fn tag(&self) -> &Option<T> {
        match self {
            StageConfig::Standard { tag, .. } => tag,
            StageConfig::Custom { tag, .. } => tag,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tag().is_none()
    }

    /// Returns true if this stage config supports dynamic resizing.
    pub fn supports_dynamic_resizing(&self) -> bool {
        match self {
            StageConfig::Standard { .. } => false, // Standard stages don't support resizing yet
            StageConfig::Custom { stage, .. } => stage.supports_dynamic_resizing(),
        }
    }

    /// Resizes this stage config to handle new dimensions.
    /// Only works for custom stages that support dynamic resizing.
    pub fn resize(&mut self, new_batch_size: usize, new_vector_dim: usize) -> Result<()> {
        match self {
            StageConfig::Standard { .. } => {
                anyhow::bail!("Standard stages do not support dynamic resizing")
            }
            StageConfig::Custom { stage, .. } => stage.resize(new_batch_size, new_vector_dim),
        }
    }

    /// Updates the FFT size parameter for custom stages that need it.
    pub fn update_n(&mut self, new_n: usize) -> Result<()> {
        match self {
            StageConfig::Standard { .. } => {
                // Standard stages don't have n parameter
                Ok(())
            }
            StageConfig::Custom { stage, .. } => stage.update_n(new_n),
        }
    }

    /// Updates the actual total elements for all stages.
    pub fn update_actual_total_elements(&mut self, total_elements: usize) -> Result<()> {
        match self {
            StageConfig::Standard { .. } => {
                // Standard stages don't have this parameter
                Ok(())
            }
            StageConfig::Custom { stage, .. } => stage.update_actual_total_elements(total_elements),
        }
    }
}

impl<T: Clone> Clone for StageConfig<T> {
    fn clone(&self) -> Self {
        match self {
            StageConfig::Standard { stage, tag } => StageConfig::Standard {
                stage: stage.clone(),
                tag: tag.clone(),
            },
            StageConfig::Custom { .. } => {
                // Custom stages cannot be cloned (they may contain non-Clone data)
                // This is a limitation - in practice, stages should be created fresh
                panic!("Cannot clone StageConfig::Custom - create a new instance instead");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestStage {
        name: String,
        vector_dim: usize,
        batch_size: usize,
    }

    impl PipelineStage for TestStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn vector_dim(&self) -> usize {
            self.vector_dim
        }

        fn batch_size(&self) -> usize {
            self.batch_size
        }

        fn encode(
            &self,
            _encoder: &mut CommandEncoder,
            _input: &wgpu::Buffer,
            _output: &wgpu::Buffer,
            _side_inputs: &HashMap<String, Arc<wgpu::Buffer>>,
        ) -> Result<()> {
            Ok(())
        }

        fn initialize(&mut self, _context: &ComputeContext) -> Result<()> {
            Ok(())
        }

        fn requires_initialization(&self) -> bool {
            false
        }
    }

    impl Debug for TestStage {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TestStage")
                .field("name", &self.name)
                .field("vector_dim", &self.vector_dim)
                .field("batch_size", &self.batch_size)
                .finish()
        }
    }

    #[test]
    fn test_stage_config_standard() {
        let stage = crate::pipeline::Stage::new("test", "wgsl", 2, 1024);
        let config: StageConfig<u64> = StageConfig::Standard { stage, tag: None };

        assert_eq!(config.name(), "test");
        assert_eq!(config.vector_dim(), 2);
        assert_eq!(config.batch_size(), 1024);
        assert!(!config.is_custom());
    }

    #[test]
    fn test_stage_config_custom() {
        let stage = Box::new(TestStage {
            name: "custom_test".to_string(),
            vector_dim: 2,
            batch_size: 2048,
        });
        let config: StageConfig<u64> = StageConfig::Custom { stage, tag: None };

        assert_eq!(config.name(), "custom_test");
        assert_eq!(config.vector_dim(), 2);
        assert_eq!(config.batch_size(), 2048);
        assert!(config.is_custom());
    }

    #[test]
    fn test_stage_config_forward_tag() {
        let stage = crate::pipeline::Stage::new("test", "wgsl", 2, 1024);
        let mut config = StageConfig::<u64>::Standard {
            stage,
            tag: Some(42),
        };

        // forward_tag should exchange the new tag with the old one
        let old_tag = config.forward_tag(Some(100));
        assert_eq!(old_tag, Some(42));
        assert_eq!(config.tag(), &Some(100));

        // Test with None
        let stage2 = crate::pipeline::Stage::new("test2", "wgsl", 2, 1024);
        let mut config2 = StageConfig::<u64>::Standard {
            stage: stage2,
            tag: None,
        };
        let old_tag2 = config2.forward_tag(Some(200));
        assert_eq!(old_tag2, None);
        assert_eq!(config2.tag(), &Some(200));
    }

    #[test]
    fn test_stage_config_forward_tag_custom() {
        let stage = Box::new(TestStage {
            name: "custom_test".to_string(),
            vector_dim: 2,
            batch_size: 2048,
        });
        let mut config = StageConfig::<String>::Custom {
            stage,
            tag: Some("initial".to_string()),
        };

        let old_tag = config.forward_tag(Some("new".to_string()));
        assert_eq!(old_tag, Some("initial".to_string()));

        // Verify new tag is stored
        match &config {
            StageConfig::Custom { tag, .. } => {
                assert_eq!(tag, &Some("new".to_string()));
            }
            _ => panic!("Expected Custom variant"),
        }
    }
}
